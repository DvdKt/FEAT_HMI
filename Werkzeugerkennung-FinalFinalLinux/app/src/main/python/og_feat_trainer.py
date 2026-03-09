"""
OG_FEAT head-only fine-tuning for the local training lists.

This keeps the Res12 backbone frozen and only adapts the FEAT attention head.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from og_feat_utils import (
    build_og_feat_model,
    get_default_init_weights_path,
    load_array_tensor,
    load_og_feat_weights,
    use_init_weights,
)

DEFAULT_HEAD_EPOCHS = 50
DEFAULT_HEAD_LR = 1e-3
DEFAULT_HEAD_BALANCE = 0.0
MODEL_FILENAME = "og_feat_model.pt"


def _get_head_epochs() -> int:
    raw = os.getenv("OG_FEAT_HEAD_EPOCHS", "").strip()
    if raw == "":
        return DEFAULT_HEAD_EPOCHS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("OG_FEAT_HEAD_EPOCHS must be an integer.") from exc
    return max(1, value)


def _get_head_lr() -> float:
    raw = os.getenv("OG_FEAT_HEAD_LR", "").strip()
    if raw == "":
        return DEFAULT_HEAD_LR
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("OG_FEAT_HEAD_LR must be a float.") from exc
    return value


def _get_head_balance() -> float:
    raw = os.getenv("OG_FEAT_HEAD_BALANCE", "").strip()
    if raw == "":
        return DEFAULT_HEAD_BALANCE
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("OG_FEAT_HEAD_BALANCE must be a float.") from exc
    return max(0.0, value)


def _split_training_shots(training: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    shots = training.get("shots", []) or []
    clean_paths = [shot.get("array_path") for shot in shots if shot.get("env_code") == "Clean"]
    non_clean_paths = [shot.get("array_path") for shot in shots if shot.get("env_code") != "Clean"]
    post_training = list(training.get("post_training_arrays", []) or [])
    clean_paths = [path for path in clean_paths if path]
    non_clean_paths = [path for path in non_clean_paths if path]
    post_training = [path for path in post_training if path]
    return clean_paths, non_clean_paths, post_training


def _interleave_by_index(groups: List[List[torch.Tensor]], count: int) -> List[torch.Tensor]:
    """
    Interleave tensors as expected by OG_FEAT indexing:
    [class0_item0, class1_item0, ..., classN_item0, class0_item1, ...].
    """
    ordered: List[torch.Tensor] = []
    for idx in range(count):
        for class_group in groups:
            ordered.append(class_group[idx])
    return ordered


def _build_episode(
    training_lists: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], int, int]:
    support_groups: List[List[torch.Tensor]] = []
    query_groups: List[List[torch.Tensor]] = []
    object_names: List[str] = []

    for training in training_lists:
        object_name = training.get("object_name", "")
        clean_paths, non_clean_paths, post_training = _split_training_shots(training)
        if not clean_paths:
            raise ValueError(f"Object {object_name} is missing Clean shots.")
        query_paths = non_clean_paths + post_training
        if not query_paths:
            raise ValueError(f"Object {object_name} has no query shots.")

        support_groups.append([load_array_tensor(path) for path in sorted(clean_paths)])
        query_groups.append([load_array_tensor(path) for path in sorted(query_paths)])
        object_names.append(object_name)

    shot = min(len(group) for group in support_groups)
    query = min(len(group) for group in query_groups)
    if shot == 0 or query == 0:
        raise ValueError("At least 1 support and 1 query per class are required.")

    # Trim to the shared count so every class contributes evenly.
    support_groups = [group[:shot] for group in support_groups]
    query_groups = [group[:query] for group in query_groups]

    # OG_FEAT expects interleaved ordering: all classes for shot 0, then shot 1, etc.
    support_ordered = _interleave_by_index(support_groups, shot)
    query_ordered = _interleave_by_index(query_groups, query)
    data = torch.stack(support_ordered + query_ordered, dim=0).unsqueeze(0)

    labels = torch.arange(len(object_names)).repeat(query)
    labels_aux = torch.arange(len(object_names)).repeat(shot + query)
    return data, labels, labels_aux, object_names, shot, query


def train_og_feat_head(base_dir: Path, training_lists: List[Dict[str, Any]]) -> Path:
    """
    Fine-tune the OG_FEAT attention head on current training lists.
    """
    if not training_lists:
        raise ValueError("No training lists provided.")

    data, labels, labels_aux, object_names, shot, query = _build_episode(training_lists)
    way = len(object_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _args = build_og_feat_model(way=way, shot=shot, query=query)
    weights_path: Path | None = None
    if use_init_weights():
        weights_path = Path(os.getenv("OG_FEAT_INIT_WEIGHTS", "")) if os.getenv("OG_FEAT_INIT_WEIGHTS") else None
        if weights_path is None or str(weights_path).strip() == "":
            weights_path = get_default_init_weights_path()
        load_og_feat_weights(model, weights_path)

    # If a prior head exists, warm-start from it.
    model_path = base_dir / "training" / MODEL_FILENAME
    if model_path.exists():
        payload = torch.load(model_path, map_location="cpu")
        state = payload.get("state_dict") if isinstance(payload, dict) else None
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)

    # Freeze the backbone to keep it stable for incremental updates.
    for param in model.encoder.parameters():
        param.requires_grad = False

    model = model.to(device)
    data = data.to(device)
    labels = labels.to(device)
    labels_aux = labels_aux.to(device)

    head_params = [p for p in model.parameters() if p.requires_grad]
    if not head_params:
        raise ValueError("No trainable parameters found in OG_FEAT head.")

    optimizer = torch.optim.Adam(head_params, lr=_get_head_lr())
    epochs = _get_head_epochs()
    balance = _get_head_balance()

    model.train()
    for _ in range(epochs):
        # Keep encoder in eval to avoid BN drift while head adapts.
        model.encoder.eval()
        optimizer.zero_grad()
        logits, reg_logits = model(data)
        loss = F.cross_entropy(logits, labels)
        if balance > 0 and reg_logits is not None:
            loss = loss + balance * F.cross_entropy(reg_logits, labels_aux)
        loss.backward()
        optimizer.step()

    save_path = base_dir / "training" / MODEL_FILENAME
    payload = {
        "version": 1,
        "backbone": "Res12",
        "way": way,
        "shot": shot,
        "query": query,
        "object_names": object_names,
        "state_dict": model.state_dict(),
        "init_weights": str(weights_path) if weights_path else None,
        "head_only": True,
    }
    torch.save(payload, save_path)
    return save_path

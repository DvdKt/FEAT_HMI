"""
OG_FEAT inference utilities used by the Mac backend server.

This module uses the OG_FEAT Res12 backbone with a fine-tuned FEAT head
and applies decision thresholds on top of embeddings.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent
PY_BACKEND_DIR = ROOT_DIR / "app" / "src" / "main" / "python"
if str(PY_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(PY_BACKEND_DIR))

from og_feat_utils import (  # noqa: E402  # isort: skip
    build_og_feat_model,
    get_default_init_weights_path,
    load_array_tensor,
    load_og_feat_weights,
    use_init_weights,
)
MODEL_FILENAME = "og_feat_model.pt"
CACHE_FILENAME = "og_feat_support_cache.pt"
CLASS_MAP_FILENAME = "class_map.json"

CACHE_VERSION = 4
CLASS_MAP_VERSION = 1


def _error(code: str, message: str, details: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}


@dataclass(frozen=True)
class ThresholdConfig:
    """
    Decision threshold configuration.

    conf_threshold: minimum confidence to accept a match.
    margin_threshold: minimum gap between best and runner-up.
    """

    conf_threshold: float
    margin_threshold: float


def _load_model(model_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load the OG_FEAT model (Res12 + FEAT head).

    If a fine-tuned checkpoint exists, it is loaded on top of the init weights.
    """
    payload: Dict[str, Any] = {}
    if model_path.exists():
        payload = torch.load(model_path, map_location="cpu")

    way = int(payload.get("way", 1)) if isinstance(payload, dict) else 1
    shot = int(payload.get("shot", 1)) if isinstance(payload, dict) else 1
    query = int(payload.get("query", 1)) if isinstance(payload, dict) else 1

    model, _args = build_og_feat_model(way=way, shot=shot, query=query)

    init_meta: Dict[str, Any] = {}
    if use_init_weights():
        weights_path = os.getenv("OG_FEAT_INIT_WEIGHTS", "").strip()
        if weights_path:
            init_weights = Path(weights_path).expanduser().resolve()
        else:
            init_weights = get_default_init_weights_path()
        init_meta = load_og_feat_weights(model, init_weights)

    if isinstance(payload, dict) and "state_dict" in payload:
        model.load_state_dict(payload["state_dict"], strict=False)

    model.eval()
    return model, {"payload": payload, "init": init_meta}


def _safe_json_load(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception as exc:  # pylint: disable=broad-except
        return {"error": str(exc)}


def _training_list_paths(base_dir: Path) -> List[Path]:
    return sorted((base_dir / "training").glob("Training_*.json"))


def _latest_training_list_mtime(base_dir: Path) -> float:
    paths = _training_list_paths(base_dir)
    if not paths:
        return 0.0
    return max(path.stat().st_mtime for path in paths)


def _load_training_lists_with_meta(base_dir: Path) -> List[Dict[str, Any]]:
    lists: List[Dict[str, Any]] = []
    for path in _training_list_paths(base_dir):
        data = _safe_json_load(path.read_text(encoding="utf-8"))
        object_name = ""
        if isinstance(data, dict):
            object_name = data.get("object_name") or ""
        lists.append(
            {
                "path": path,
                "mtime": path.stat().st_mtime,
                "object_name": object_name,
                "data": data,
            }
        )
    return lists


def _current_training_list_mtimes(base_dir: Path) -> Dict[str, float]:
    mtimes: Dict[str, float] = {}
    for item in _load_training_lists_with_meta(base_dir):
        object_name = item.get("object_name") or ""
        if not object_name:
            continue
        mtimes[object_name] = float(item.get("mtime") or 0.0)
    return mtimes


def _collect_support_paths(training: Dict[str, Any]) -> List[str]:
    support_paths: List[str] = []
    for shot in training.get("shots", []) or []:
        path = shot.get("array_path")
        if path:
            support_paths.append(path)
    for path in training.get("post_training_arrays", []) or []:
        if path:
            support_paths.append(path)
    return support_paths


def _load_class_map(base_dir: Path) -> Dict[str, Any]:
    path = base_dir / "training" / CLASS_MAP_FILENAME
    if not path.exists():
        return {"version": CLASS_MAP_VERSION, "next_id": 0, "name_to_id": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pylint: disable=broad-except
        return {"version": CLASS_MAP_VERSION, "next_id": 0, "name_to_id": {}}
    if int(payload.get("version", 0)) != CLASS_MAP_VERSION:
        return {"version": CLASS_MAP_VERSION, "next_id": 0, "name_to_id": {}}
    return payload


def _save_class_map(base_dir: Path, payload: Dict[str, Any]) -> None:
    path = base_dir / "training" / CLASS_MAP_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ensure_class_map(base_dir: Path, object_names: List[str]) -> Dict[str, int]:
    payload = _load_class_map(base_dir)
    name_to_id = payload.get("name_to_id", {}) or {}
    next_id = int(payload.get("next_id", 0))

    for name in object_names:
        if name not in name_to_id:
            name_to_id[name] = next_id
            next_id += 1

    payload.update(
        {
            "version": CLASS_MAP_VERSION,
            "next_id": next_id,
            "name_to_id": name_to_id,
            "active_names": object_names,
        }
    )
    _save_class_map(base_dir, payload)
    return {str(k): int(v) for k, v in name_to_id.items()}


def _build_support_cache(
    base_dir: Path,
    model: torch.nn.Module,
    training_lists: List[Dict[str, Any]],
    previous_cache: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build per-class prototypes from Training_*.json.
    """
    object_names: List[str] = []
    prototypes: List[torch.Tensor] = []
    training_list_mtimes: Dict[str, float] = {}

    prev_proto_by_name: Dict[str, torch.Tensor] = {}
    prev_mtimes: Dict[str, float] = {}
    if previous_cache:
        prev_names = previous_cache.get("object_names") or []
        prev_proto = previous_cache.get("prototypes")
        if prev_proto is not None:
            if not isinstance(prev_proto, torch.Tensor):
                prev_proto = torch.tensor(prev_proto, dtype=torch.float32)
            for idx, name in enumerate(prev_names):
                if idx < prev_proto.size(0):
                    prev_proto_by_name[name] = prev_proto[idx : idx + 1].clone()
        prev_mtimes = previous_cache.get("training_list_mtimes", {}) or {}

    with torch.no_grad():
        for meta in training_lists:
            training = meta.get("data") or {}
            object_name = meta.get("object_name") or ""
            support_paths = _collect_support_paths(training) if isinstance(training, dict) else []
            if not object_name or not support_paths:
                continue

            mtime = float(meta.get("mtime") or 0.0)
            proto_tensor: Optional[torch.Tensor] = None

            reuse = (
                object_name in prev_proto_by_name
                and prev_mtimes.get(object_name) == mtime
            )
            if reuse:
                proto_tensor = prev_proto_by_name[object_name]

            if proto_tensor is None:
                embeddings: List[np.ndarray] = []
                for path in support_paths:
                    tensor = load_array_tensor(path).unsqueeze(0)
                    feat = model.encoder(tensor)
                    if feat.dim() > 2:
                        feat = feat.view(feat.size(0), -1)
                    embeddings.append(feat.squeeze(0).cpu().numpy())

                if not embeddings:
                    continue

                mean_vec = np.mean(np.stack(embeddings, axis=0), axis=0)
                proto_tensor = torch.tensor(mean_vec, dtype=torch.float32).unsqueeze(0)

            object_names.append(object_name)
            prototypes.append(proto_tensor)
            training_list_mtimes[object_name] = mtime

        if not prototypes:
            return _error("NO_SUPPORT", "No support arrays available for inference.", {})

        proto = torch.cat(prototypes, dim=0)
        proto_in = proto.unsqueeze(0)
        # Adapt class prototypes with the FEAT attention head.
        adapted_proto = model.slf_attn(proto_in, proto_in, proto_in).squeeze(0)

    class_map = _ensure_class_map(base_dir, object_names)

    return {
        "version": CACHE_VERSION,
        "object_names": object_names,
        "prototypes": proto,
        "adapted_prototypes": adapted_proto,
        "class_map": class_map,
        "training_list_mtimes": training_list_mtimes,
    }


def refresh_support_cache(base_dir: Path) -> Dict[str, Any]:
    """
    Recompute the support/prototype cache from current training lists.
    """
    model_path = base_dir / "training" / MODEL_FILENAME
    try:
        model, _payload = _load_model(model_path)
    except Exception as exc:  # pylint: disable=broad-except
        return _error("MODEL_LOAD_FAILED", "OG_FEAT model not found for cache refresh.", {"error": str(exc)})

    training_lists = _load_training_lists_with_meta(base_dir)
    if not training_lists:
        return _error("NO_TRAINING_LISTS", "No training lists found.", {})

    previous_cache = _load_support_cache_raw(base_dir)
    if previous_cache is not None:
        if int(previous_cache.get("version", 0)) != CACHE_VERSION:
            previous_cache = None
        elif previous_cache.get("model_mtime") != model_path.stat().st_mtime:
            previous_cache = None

    cache = _build_support_cache(
        base_dir,
        model,
        training_lists,
        previous_cache,
    )
    if "error" in cache:
        return cache

    cache.update(
        {
            "model_mtime": model_path.stat().st_mtime if model_path.exists() else 0.0,
            "training_lists_mtime": _latest_training_list_mtime(base_dir),
        }
    )
    cache_path = base_dir / "training" / CACHE_FILENAME
    torch.save(cache, cache_path)
    return {"cached": True, "cache_path": str(cache_path), "object_count": len(cache["object_names"])}


def _load_support_cache_raw(base_dir: Path) -> Optional[Dict[str, Any]]:
    cache_path = base_dir / "training" / CACHE_FILENAME
    if not cache_path.exists():
        return None
    try:
        return torch.load(cache_path, map_location="cpu")
    except Exception:  # pylint: disable=broad-except
        return None


def _load_support_cache(base_dir: Path) -> Optional[Dict[str, Any]]:
    cache_path = base_dir / "training" / CACHE_FILENAME
    if not cache_path.exists():
        return None
    cache = _load_support_cache_raw(base_dir)
    if cache is None:
        return None

    if int(cache.get("version", 0)) != CACHE_VERSION:
        return None

    model_path = base_dir / "training" / MODEL_FILENAME
    if model_path.exists() and cache.get("model_mtime") != model_path.stat().st_mtime:
        return None

    cached_mtimes = cache.get("training_list_mtimes")
    if isinstance(cached_mtimes, dict):
        current_mtimes = _current_training_list_mtimes(base_dir)
        if cached_mtimes != current_mtimes:
            return None
    elif cache.get("training_lists_mtime") != _latest_training_list_mtime(base_dir):
        return None
    return cache


def run_feat_inference(
    base_dir: Path,
    array_path: Path,
    thresholds: ThresholdConfig,
) -> Dict[str, Any]:
    """
    Run OG_FEAT inference on a single query array and return probabilities + thresholds.
    """
    model_path = base_dir / "training" / MODEL_FILENAME
    try:
        model, _payload = _load_model(model_path)
    except Exception as exc:  # pylint: disable=broad-except
        return _error("MODEL_LOAD_FAILED", "Failed to load OG_FEAT model.", {"error": str(exc)})

    cache = _load_support_cache(base_dir)
    if cache is None:
        refresh = refresh_support_cache(base_dir)
        if "error" in refresh:
            return refresh
        cache = _load_support_cache(base_dir)
    if cache is None:
        return _error("SUPPORT_CACHE_FAILED", "Failed to load support cache.", {})

    object_names = cache.get("object_names") or []
    if not object_names:
        return _error(
            "INSUFFICIENT_CLASSES",
            "At least 1 class is required for inference.",
            {"class_count": len(object_names)},
        )

    adapted_proto = cache.get("adapted_prototypes")
    if adapted_proto is None:
        return _error("SUPPORT_CACHE_FAILED", "Support cache is missing prototypes.", {})

    try:
        query = load_array_tensor(str(array_path)).unsqueeze(0)
    except Exception as exc:  # pylint: disable=broad-except
        return _error("INFERENCE_INPUT_FAILED", "Failed to build inference tensors.", {"error": str(exc)})

    with torch.no_grad():
        q_feat = model.encoder(query)
        if q_feat.dim() > 2:
            q_feat = q_feat.view(q_feat.size(0), -1)
        if not isinstance(adapted_proto, torch.Tensor):
            adapted_proto = torch.tensor(adapted_proto)
        n, m, d = q_feat.size(0), adapted_proto.size(0), q_feat.size(1)
        x = q_feat.unsqueeze(1).expand(n, m, d)
        y = adapted_proto.unsqueeze(0).expand(n, m, d)
        logits = -torch.pow(x - y, 2).sum(2)
        temperature = getattr(getattr(model, "args", None), "temperature", 1.0)
        if float(temperature) != 0.0:
            logits = logits / float(temperature)

    logits_list = logits.squeeze(0).detach().cpu().numpy().tolist()
    query_embedding = q_feat.squeeze(0).detach().cpu().numpy()
    logits_array = np.array(logits_list, dtype=np.float32)
    if logits_array.size:
        max_logit = float(np.max(logits_array))
        exp_logits = np.exp(logits_array - max_logit)
        feat_probs = exp_logits / np.sum(exp_logits)
        feat_confidence = float(np.max(feat_probs))
    else:
        feat_confidence = 0.0

    prob_values = feat_probs.tolist()
    ranked = sorted(enumerate(prob_values), key=lambda item: item[1], reverse=True)
    top_k = ranked[: max(2, min(5, len(ranked)))]
    max_prob = float(prob_values[top_k[0][0]]) if top_k else 0.0
    second_prob = float(prob_values[top_k[1][0]]) if len(top_k) > 1 else 0.0
    margin = max_prob - second_prob
    passed_conf = max_prob >= thresholds.conf_threshold
    passed_margin = True
    if thresholds.margin_threshold > 0:
        passed_margin = margin >= thresholds.margin_threshold
    accepted = passed_conf and passed_margin
    predicted_label = object_names[top_k[0][0]] if top_k else None
    runner_up_label = object_names[top_k[1][0]] if len(top_k) > 1 else None
    unknown_prob = max(0.0, 1.0 - max_prob)

    class_map = cache.get("class_map", {}) or {}
    probs_topk = [
        {
            "object_name": object_names[idx],
            "class_id": class_map.get(object_names[idx]),
            "prob": float(prob),
        }
        for idx, prob in top_k
    ]

    predicted_class_id = class_map.get(predicted_label) if predicted_label else None
    runner_up_class_id = class_map.get(runner_up_label) if runner_up_label else None

    result = {
        "probs_topk": probs_topk,
        "max_prob": max_prob,
        "second_prob": second_prob,
        "margin": margin,
        "passed_conf": passed_conf,
        "passed_margin": passed_margin,
        "accepted_by_threshold": accepted,
        "is_unknown": not accepted,
        "predicted_label": predicted_label,
        "predicted_class_id": predicted_class_id,
        "unknown_prob": unknown_prob,
        "best_known_prob": max_prob,
        "runner_up_label": runner_up_label,
        "runner_up_class_id": runner_up_class_id,
        "runner_up_prob": second_prob,
        "feat_confidence": feat_confidence,
    }
    return result

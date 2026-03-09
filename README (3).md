"""
Shared OG_FEAT helpers for model loading and preprocessing.

This module keeps all OG_FEAT-specific assumptions (paths, preprocessing,
default weights) in one place so training and inference stay consistent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

DEFAULT_BACKBONE = "Res12"
DEFAULT_IMAGE_SIZE = 84
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TEMPERATURE2 = 1.0
DEFAULT_USE_INIT_WEIGHTS = False

# OG_FEAT Res12 normalization (same as OG_FEAT dataloaders for Res12).
DEFAULT_MEAN = (120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0)
DEFAULT_STD = (70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0)


def _get_repo_root() -> Path:
    """Resolve repo root from app/src/main/python."""
    return Path(__file__).resolve().parents[4]


def get_og_feat_root() -> Path:
    """
    Locate OG_FEAT folder.

    Uses OG_FEAT_ROOT if set, otherwise assumes it lives at <repo>/OG_FEAT.
    """
    raw = os.getenv("OG_FEAT_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _get_repo_root() / "OG_FEAT"


def ensure_og_feat_on_path() -> Path:
    """
    Ensure OG_FEAT root is on sys.path so `import model.*` resolves.
    """
    root = get_og_feat_root()
    if not root.exists():
        raise FileNotFoundError(f"OG_FEAT folder not found at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def get_default_init_weights_path() -> Path:
    """
    Default TieredImageNet + Res12 init weights path from OG_FEAT.
    """
    root = get_og_feat_root()
    return root / "saves" / "initialization" / "tieredimagenet" / "Res12-pre.pth"


def _get_image_size() -> int:
    raw = os.getenv("OG_FEAT_IMAGE_SIZE", "").strip()
    if raw == "":
        return DEFAULT_IMAGE_SIZE
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("OG_FEAT_IMAGE_SIZE must be an integer.") from exc
    if value <= 0:
        raise ValueError("OG_FEAT_IMAGE_SIZE must be > 0.")
    return value


def _get_temperature() -> float:
    raw = os.getenv("OG_FEAT_TEMPERATURE", "").strip()
    if raw == "":
        return DEFAULT_TEMPERATURE
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError("OG_FEAT_TEMPERATURE must be a float.") from exc


def _get_temperature2() -> float:
    raw = os.getenv("OG_FEAT_TEMPERATURE2", "").strip()
    if raw == "":
        return DEFAULT_TEMPERATURE2
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError("OG_FEAT_TEMPERATURE2 must be a float.") from exc


def use_init_weights() -> bool:
    raw = os.getenv("OG_FEAT_USE_INIT_WEIGHTS", "").strip().lower()
    if raw == "":
        return DEFAULT_USE_INIT_WEIGHTS
    return raw in {"1", "true", "yes", "y", "on"}


def build_og_feat_args(
    way: int,
    shot: int,
    query: int,
    *,
    eval_way: Optional[int] = None,
    eval_shot: Optional[int] = None,
    eval_query: Optional[int] = None,
    use_euclidean: bool = True,
) -> SimpleNamespace:
    """
    Build a minimal args namespace for OG_FEAT model construction.
    """
    return SimpleNamespace(
        backbone_class=DEFAULT_BACKBONE,
        way=way,
        shot=shot,
        query=query,
        eval_way=eval_way if eval_way is not None else way,
        eval_shot=eval_shot if eval_shot is not None else shot,
        eval_query=eval_query if eval_query is not None else query,
        use_euclidean=use_euclidean,
        temperature=_get_temperature(),
        temperature2=_get_temperature2(),
        balance=0.0,
    )


def build_og_feat_model(
    way: int,
    shot: int,
    query: int,
    *,
    eval_way: Optional[int] = None,
    eval_shot: Optional[int] = None,
    eval_query: Optional[int] = None,
    use_euclidean: bool = True,
) -> Tuple[torch.nn.Module, SimpleNamespace]:
    """
    Construct an OG_FEAT model with Res12 backbone.
    """
    ensure_og_feat_on_path()
    from model.models.feat import FEAT  # type: ignore

    args = build_og_feat_args(
        way,
        shot,
        query,
        eval_way=eval_way,
        eval_shot=eval_shot,
        eval_query=eval_query,
        use_euclidean=use_euclidean,
    )
    model = FEAT(args)
    return model, args


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    """Remove 'module.' prefix added by DataParallel."""
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def load_og_feat_weights(model: torch.nn.Module, weights_path: Path) -> Dict[str, Any]:
    """
    Load a checkpoint into an OG_FEAT model.

    Supports both encoder-only and full-model checkpoints.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"OG_FEAT weights not found at {weights_path}")
    payload = torch.load(weights_path, map_location="cpu")
    if isinstance(payload, dict) and "params" in payload:
        state = payload["params"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise ValueError("Unsupported OG_FEAT checkpoint format.")

    state = _strip_module_prefix(state)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(filtered)
    model.load_state_dict(model_state)
    return {
        "loaded_keys": len(filtered),
        "total_keys": len(model_state),
        "weights_path": str(weights_path),
    }


def _resize_array(array: np.ndarray, size: int) -> np.ndarray:
    if array.shape[0] == size and array.shape[1] == size:
        return array
    image = Image.fromarray(array)
    image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def load_array_tensor(path: str) -> torch.Tensor:
    """
    Load an array from disk and apply OG_FEAT preprocessing.
    """
    array = np.load(path)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Array at {path} must have shape (H, W, 3).")
    size = _get_image_size()
    array = _resize_array(array, size)
    if not array.flags.writeable:
        array = array.copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(DEFAULT_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(DEFAULT_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std

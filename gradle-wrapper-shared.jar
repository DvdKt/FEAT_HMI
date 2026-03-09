"""
Public API surface for the Python backend.

Integration style:
- Local module API with JSON-serializable dict returns.
- GUI/Kotlin layer should call these functions directly (no CLI prompts).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from constants import ENVIRONMENT_SPEC
from core import Backend, BackendConfig, _error

_BACKEND: Optional[Backend] = None


def _extract_sequence_from_path(path_str: str) -> Optional[int]:
    """Extract numeric sequence suffix from a filename like name_ENV_0001.npy."""
    if not path_str:
        return None
    stem = os.path.splitext(os.path.basename(path_str))[0]
    parts = stem.split("_")
    if not parts:
        return None
    candidate = parts[-1]
    return int(candidate) if candidate.isdigit() else None


def _is_complete_training_list(training: Dict[str, Any]) -> bool:
    """
    Return True when a training list has the required 5-shot structure.

    Phase 2 can create partial training lists containing only PostTraining arrays,
    so FEAT training must skip incomplete lists to stay robust.
    """
    shots = training.get("shots", [])
    if len(shots) != 5:
        return False
    clean = [shot for shot in shots if shot.get("env_code") == "Clean" and shot.get("array_path")]
    non_clean = [shot for shot in shots if shot.get("env_code") != "Clean" and shot.get("array_path")]
    return len(clean) == 2 and len(non_clean) == 3


def _training_file_to_ui(training_file: Path, object_id_by_name: Dict[str, str]) -> Dict[str, Any]:
    with training_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    object_name = data.get("object_name", "")
    shots = []
    for shot in data.get("shots", []):
        shots.append(
            {
                "env_code": shot.get("env_code", ""),
                "sequence": _extract_sequence_from_path(shot.get("array_path", "")),
            }
        )
    post_training = []
    for path in data.get("post_training_arrays", []) or []:
        post_training.append(
            {
                "env_code": "PostTraining",
                "sequence": _extract_sequence_from_path(path),
            }
        )
    return {
        "object_id": object_id_by_name.get(object_name, ""),
        "training_file": str(training_file),
        "shots": shots,
        "post_training_shots": post_training,
    }


def get_env_spec() -> List[Dict[str, Any]]:
    """
    Return environment spec in the shape expected by the Android UI.

    Example:
        get_env_spec() -> [{"env_code": "BG1", "friendly_name": "...", "is_clean": false}]
    """
    return [
        {
            "env_code": env["code"],
            "env_name": env["category"],
            "friendly_name": env["name"],
            "is_clean": env["code"] == "Clean",
        }
        for env in ENVIRONMENT_SPEC
    ]


def initialize_backend(base_dir: str) -> Dict[str, Any]:
    """
    Initialize the backend and base directory layout.

    Example:
        initialize_backend("/tmp/feat_project") -> {"ok": True, "base_dir": "...", "state": "..."}
    """
    global _BACKEND  # noqa: PLW0603 - module-level singleton for Kotlin integration
    if not base_dir:
        return _error("INVALID_BASE_DIR", "base_dir is required.", {})
    _BACKEND = Backend(BackendConfig(base_dir=Path(base_dir)))
    return _BACKEND.initialize()


def create_object(object_name: str, overwrite: bool = False) -> Dict[str, Any]:
    """
    Create a new object with sanitized name and filesystem layout.

    Example:
        create_object("Hammer") -> {"object_id": "...", "name": "hammer", ...}
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    if not object_name:
        return _error("INVALID_NAME", "object_name is required.", {})
    return _BACKEND.create_object(object_name=object_name, overwrite=overwrite)


def list_objects() -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    List all object records.

    Example:
        list_objects() -> [{"object_id": "...", "name": "hammer", ...}]
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    return _BACKEND.list_objects()


def get_object(object_id: str) -> Dict[str, Any]:
    """
    Fetch a single object record by id.

    Example:
        get_object("uuid") -> {"object_id": "...", "name": "..."}
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    return _BACKEND.get_object(object_id)


def get_next_required_shot(object_id: str) -> Dict[str, Any]:
    """
    Return the next required shot for the object.

    Example:
        get_next_required_shot(obj_id) -> {"env_code": "Clean", "remaining_needed_for_env": 2, ...}
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    return _BACKEND.get_next_required_shot(object_id)


def submit_shot(
    object_id: str,
    env_code: str,
    image_base64: str,
    accept: bool,
    image_ext: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Submit a shot attempt.

    Example:
        submit_shot(obj_id, "Clean", image_b64, True)
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    return _BACKEND.submit_shot(object_id, env_code, image_base64, accept, image_ext)


def can_select_environments() -> Dict[str, Any]:
    """
    Check whether environment selection is allowed.

    Example:
        can_select_environments() -> {"ok": True, "reason": "..."}
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    return _BACKEND.can_select_environments()


def set_selected_training_environments(env_codes: List[str]) -> Dict[str, Any]:
    """
    Select which non-Clean environments should be used for training.

    Example:
        set_selected_training_environments(["BG1", "L2", "O1"])
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    return _BACKEND.set_selected_training_environments(env_codes)


def get_selected_training_environments() -> Dict[str, Any]:
    """
    Read the current environment selection.

    Example:
        get_selected_training_environments() -> {"selected_env_codes": ["BG1"], "remaining_to_select": 2}
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    return _BACKEND.get_selected_training_environments()


def select_training_environment(env_code: str, confirm: bool = True) -> Dict[str, Any]:
    """
    Confirm one environment selection (one-by-one flow).

    Example:
        select_training_environment("BG1", confirm=True)
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    return _BACKEND.add_selected_training_environment(env_code, confirm)


def build_training_sets() -> Dict[str, Any]:
    """
    Build training JSON files and FEAT export manifests.

    Example:
        build_training_sets() -> {"built": True, "training_files": [".../Training_obj.json"]}
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    return _BACKEND.build_training_sets()


def build_training_sets_for_ui() -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build training sets and return a UI-friendly list of training outputs.

    Example:
        build_training_sets_for_ui() -> [{"training_file": "...", "shots": [...]}]
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})

    result = _BACKEND.build_training_sets()
    if "error" in result:
        return result

    training_files = result.get("training_files", [])
    object_id_by_name = {obj["name"]: obj["object_id"] for obj in _BACKEND.state.get("objects", [])}
    training_lists: List[Dict[str, Any]] = []
    for training_file in training_files:
        training_lists.append(_training_file_to_ui(Path(training_file), object_id_by_name))
    return training_lists


def list_training_sets() -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    List existing training lists without rebuilding them.

    Example:
        list_training_sets() -> [{"training_file": "...", "shots": [...]}]
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})

    object_id_by_name = {obj["name"]: obj["object_id"] for obj in _BACKEND.state.get("objects", [])}
    training_dir = _BACKEND.base_dir / "training"
    training_lists: List[Dict[str, Any]] = []
    for training_file in sorted(training_dir.glob("Training_*.json")):
        training_lists.append(_training_file_to_ui(training_file, object_id_by_name))
    return training_lists


def reset_all() -> Dict[str, Any]:
    """
    Delete all objects and reset backend state to defaults.

    Example:
        reset_all() -> {"ok": True}
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    return _BACKEND.reset_all()


def mark_inference_started() -> Dict[str, Any]:
    """
    Mark the beginning of in-the-wild inference (baseline snapshot).
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    return _BACKEND.mark_inference_started()


def get_soft_reset_status() -> Dict[str, Any]:
    """
    Return readiness status for soft reset.
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})

    state = _BACKEND.state
    selected_envs = state.get("selected_envs") or []
    envs_selected = len(selected_envs) == 3
    objects = state.get("objects") or []
    objects_created = len(objects) > 0
    inference_started = bool(state.get("inference_started", False))
    baseline_ids = state.get("baseline_object_ids") or []
    baseline_objects = [obj for obj in objects if obj.get("object_id") in baseline_ids]
    baseline_count = len(baseline_objects)

    training_lists_complete = False
    missing_lists: List[str] = []
    if baseline_objects:
        training_lists_complete = True
        training_dir = _BACKEND.base_dir / "training"
        for obj in baseline_objects:
            name = obj.get("name") or ""
            training_path = training_dir / f"Training_{name}.json"
            if not training_path.exists():
                training_lists_complete = False
                missing_lists.append(name)
                continue
            with training_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not _is_complete_training_list(data):
                training_lists_complete = False
                missing_lists.append(name)

    return {
        "envs_selected": envs_selected,
        "objects_created": objects_created,
        "training_lists_complete": training_lists_complete,
        "baseline_objects": baseline_count,
        "inference_started": inference_started,
        "missing_training_lists": missing_lists,
    }


def soft_reset_to_baseline() -> Dict[str, Any]:
    """
    Revert to the baseline object set and remove Phase 2 artifacts.
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    return _BACKEND.soft_reset_to_baseline()


def train_feat_model() -> Dict[str, Any]:
    """
    Optional OG_FEAT head fine-tuning hook using manifests built by build_training_sets().

    If torch is unavailable on-device, this returns a clear error.
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})

    try:
        from og_feat_trainer import train_og_feat_head
    except Exception as exc:  # pylint: disable=broad-except
        return _error(
            "FEAT_IMPORT_ERROR",
            "OG_FEAT training dependencies are unavailable on this device.",
            {"error": str(exc)},
        )

    training_lists = []
    skipped_objects: List[str] = []
    training_dir = _BACKEND.base_dir / "training"
    for training_file in training_dir.glob("Training_*.json"):
        with open(training_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not _is_complete_training_list(data):
            object_name = data.get("object_name") if isinstance(data, dict) else None
            skipped_objects.append(object_name or training_file.name)
            continue
        training_lists.append(data)

    if not training_lists:
        return _error(
            "NO_TRAINING_LISTS",
            "No complete training lists found. Build training sets first.",
            {"skipped_objects": skipped_objects},
        )

    try:
        model_path = train_og_feat_head(_BACKEND.base_dir, training_lists)
    except Exception as exc:  # pylint: disable=broad-except
        return _error("FEAT_TRAINING_FAILED", "OG_FEAT training failed.", {"error": str(exc)})

    return {"trained": True, "model_path": str(model_path), "skipped_objects": skipped_objects}


def self_check() -> Dict[str, Any]:
    """
    Validate base_dir layout and state.json consistency.

    Example:
        self_check() -> {"ok": True, "issues": [], "state": "COLLECTING_OBJECTS"}
    """
    if _BACKEND is None:
        return _error("NOT_INITIALIZED", "Call initialize_backend first.", {})
    return _BACKEND.self_check()

"""Core backend logic for data collection and training prep."""

from __future__ import annotations

import base64
import binascii
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from constants import (
    ALL_ENV_CODES,
    ENVIRONMENT_SPEC,
    POST_TRAINING_CODE,
    REQUIRED_COUNTS,
    REQUIRED_ENV_ORDER,
    STATE_COLLECTING_OBJECTS,
    STATE_SELECTING_ENVS,
    STATE_TRAINING_BUILT,
    STATE_TRAINING_READY,
)


# Filename-safe: lower-case, digits, underscore, dash, dot.
_SAFE_NAME_RE = re.compile(r"[^a-z0-9_.-]+")
_IMAGE_LIBS: Optional[tuple[Any, Any]] = None


def _load_image_libs() -> tuple[Any, Any]:
    """Lazy-load numpy and PIL to keep backend startup fast on device."""
    global _IMAGE_LIBS  # noqa: PLW0603 - module cache for heavy imports
    if _IMAGE_LIBS is None:
        import numpy as np  # pylint: disable=import-error
        from PIL import Image  # pylint: disable=import-error

        _IMAGE_LIBS = (np, Image)
    return _IMAGE_LIBS



def _error(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a JSON-serializable error response."""
    return {"error": {"code": code, "message": message, "details": details or {}}}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _sanitize_object_name(name: str) -> str:
    """
    Sanitize object names to be Android-safe and filename-safe.

    Rules:
    - lower-case
    - replace non-alphanumeric with "_"
    - collapse multiple "_"
    - strip leading/trailing "_"
    - only allow a-z 0-9 _ - . in filenames
    """
    lowered = name.strip().lower()
    normalized = _SAFE_NAME_RE.sub("_", lowered)
    collapsed = re.sub(r"_+", "_", normalized).strip("_")
    return collapsed or "object"


def _unique_object_name(base_name: str, existing: List[str]) -> str:
    """Append _2, _3, ... when needed to avoid collisions."""
    if base_name not in existing:
        return base_name
    counter = 2
    while f"{base_name}_{counter}" in existing:
        counter += 1
    return f"{base_name}_{counter}"


def _next_required_env(
    shots: List[Dict[str, Any]],
    selected_envs: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Determine the next required environment code, in strict order.

    We treat the required order as a list of slots:
    [Clean, Clean, <selected_env_1>, <selected_env_2>, <selected_env_3>]
    Each slot is fulfilled only when the accepted count for that env_code
    exceeds the number of already-seen slots for that env_code.

    Phase 2 note:
    - We ignore any shot whose env_code is not part of ALL_ENV_CODES (e.g. PostTraining),
      so post-training samples do not affect Phase 1 completion logic.
    """
    required_order = ["Clean", "Clean"] + selected_envs
    required_counts: Dict[str, int] = {"Clean": 2, **{code: 1 for code in selected_envs}}
    counts: Dict[str, int] = {code: 0 for code in required_counts}
    for shot in shots:
        env_code = shot.get("env_code")
        if env_code in counts:
            counts[env_code] += 1

    seen_slots: Dict[str, int] = {code: 0 for code in required_counts}
    for env_code in required_order:
        if counts.get(env_code, 0) <= seen_slots.get(env_code, 0):
            remaining = required_counts[env_code] - counts.get(env_code, 0)
            return {
                "env_code": env_code,
                "remaining_needed_for_env": remaining,
                "object_completed": False,
            }
        seen_slots[env_code] += 1

    return None


def _object_completed(shots: List[Dict[str, Any]], selected_envs: List[str]) -> bool:
    if len(selected_envs) != 3:
        return False
    return _next_required_env(shots, selected_envs) is None


def _post_training_arrays(shots: List[Dict[str, Any]]) -> List[str]:
    """Collect array paths for Phase 2 PostTraining shots."""
    return [
        shot["array_path"]
        for shot in shots
        if shot.get("env_code") == POST_TRAINING_CODE
    ]


@dataclass
class BackendConfig:
    base_dir: Path


class Backend:
    """
    Backend state manager.

    Notes for maintainers:
    - The state machine is persisted in base_dir/state.json.
    - The file structure must remain stable because FEAT integration depends on it.
    - Public API returns JSON-serializable dicts only; no exceptions are exposed.
    - Global state flow (strict):
        SELECTING_ENVS -> TRAINING_READY -> TRAINING_BUILT
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.base_dir = config.base_dir
        self.state_path = self.base_dir / "state.json"
        # Phase 2 pending shots are stored under base_dir/pending/<pending_id>/.
        # We keep metadata in-memory only; pending IDs are invalid after restart.
        self.pending_dir = self.base_dir / "pending"
        self.pending_shots: Dict[str, Dict[str, Any]] = {}
        self._ensure_base_layout()
        self.state = self._load_or_init_state()

    def _ensure_base_layout(self) -> None:
        # Filesystem layout (single source of truth):
        # <base_dir>/
        #   raw/<object_name>/originals/   -> accepted original images
        #   raw/<object_name>/arrays/      -> accepted arrays (.npy)
        #   training/Training_<object>.json
        #   training/selected_envs.json
        #   training/feat_export/<object>/manifest.json
        #   pending/<pending_id>/         -> temporary Phase 2 shots awaiting confirmation
        (self.base_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "training").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "training" / "feat_export").mkdir(parents=True, exist_ok=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_init_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            state = _load_json(self.state_path)
            # Backward-compatible defaults for older state.json files.
            state.setdefault("version", 1)
            state.setdefault("global_state", STATE_SELECTING_ENVS)
            state.setdefault("next_instance_id", 1)
            state.setdefault("objects", [])
            state.setdefault("selected_envs", [])
            state.setdefault("inference_started", False)
            state.setdefault("baseline_object_ids", [])
            return state
        state = {
            "version": 1,
            "global_state": STATE_SELECTING_ENVS,
            "next_instance_id": 1,
            "objects": [],
            # selected_envs stores the user's 3 chosen non-Clean environments.
            # It starts as None (not started) and becomes a list as the user confirms picks.
            "selected_envs": [],
            "inference_started": False,
            "baseline_object_ids": [],
        }
        _save_json(self.state_path, state)
        return state

    def _persist_state(self) -> None:
        _save_json(self.state_path, self.state)

    def _find_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        for obj in self.state["objects"]:
            if obj["object_id"] == object_id:
                return obj
        return None

    def _object_dir(self, object_name: str) -> Path:
        return self.base_dir / "raw" / object_name

    def _object_originals_dir(self, object_name: str) -> Path:
        return self._object_dir(object_name) / "originals"

    def _object_arrays_dir(self, object_name: str) -> Path:
        return self._object_dir(object_name) / "arrays"

    def _next_sequence(self, obj: Dict[str, Any]) -> str:
        """Return the next per-object sequence (accepted shots only)."""
        return f"{len(obj['shots']) + 1:04d}"

    def initialize(self) -> Dict[str, Any]:
        """Return backend status with version and base_dir."""
        return {
            "ok": True,
            "base_dir": str(self.base_dir),
            "state": self.state["global_state"],
        }

    def create_object(self, object_name: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Create a new object record and its filesystem structure.

        Example:
            create_object("Hammer") -> {"object_id": "...", "name": "hammer", ...}
        """
        allowed_states = {STATE_SELECTING_ENVS, STATE_TRAINING_READY, STATE_TRAINING_BUILT}
        if self.state["global_state"] not in allowed_states:
            return _error(
                "STATE_ERROR",
                "Objects can only be created after training environments are selected.",
                {"state": self.state["global_state"]},
            )
        selected_envs = self.state.get("selected_envs") or []
        if len(selected_envs) != 3:
            return _error(
                "ENV_SELECTION_REQUIRED",
                "Select 3 environments before creating objects.",
                {"selected_env_codes": selected_envs},
            )

        sanitized = _sanitize_object_name(object_name)
        existing_names = [o["name"] for o in self.state["objects"]]

        # Overwrite means: reuse the sanitized name and delete any prior record+folder.
        if overwrite:
            unique_name = sanitized
            obj_dir = self._object_dir(unique_name)
            if obj_dir.exists():
                shutil.rmtree(obj_dir)
            self.state["objects"] = [o for o in self.state["objects"] if o["name"] != unique_name]
        else:
            unique_name = _unique_object_name(sanitized, existing_names)
            obj_dir = self._object_dir(unique_name)
            if obj_dir.exists():
                return _error(
                    "OBJECT_EXISTS",
                    "Object directory already exists. Pass overwrite=true to replace it.",
                    {"object_name": unique_name},
                )

        # Per-object folders live under raw/<object_name>/.
        self._object_originals_dir(unique_name).mkdir(parents=True, exist_ok=True)
        self._object_arrays_dir(unique_name).mkdir(parents=True, exist_ok=True)

        instance_id = f"{self.state['next_instance_id']:04d}"
        self.state["next_instance_id"] += 1

        obj = {
            "object_id": str(uuid.uuid4()),
            "name": unique_name,
            "instance_id": instance_id,
            "completed": False,
            "shots": [],
            "baseline": not bool(self.state.get("inference_started", False)),
        }
        self.state["objects"].append(obj)
        self._persist_state()
        return obj

    def mark_inference_started(self) -> Dict[str, Any]:
        """
        Mark the beginning of in-the-wild inference and freeze baseline objects once.
        """
        if not self.state.get("inference_started", False):
            self.state["inference_started"] = True
            baseline_ids = self.state.get("baseline_object_ids") or []
            if not baseline_ids:
                baseline_ids = [obj["object_id"] for obj in self.state.get("objects", [])]
                for obj in self.state.get("objects", []):
                    obj["baseline"] = True
                self.state["baseline_object_ids"] = baseline_ids
            self._persist_state()
        return {
            "inference_started": bool(self.state.get("inference_started", False)),
            "baseline_count": len(self.state.get("baseline_object_ids") or []),
        }

    def list_objects(self) -> List[Dict[str, Any]]:
        """List all object records."""
        return self.state["objects"]

    def get_object(self, object_id: str) -> Dict[str, Any]:
        """Retrieve a single object record by id."""
        obj = self._find_object(object_id)
        if not obj:
            return _error("NOT_FOUND", "Object not found.", {"object_id": object_id})
        return obj

    def get_next_required_shot(self, object_id: str) -> Dict[str, Any]:
        """
        Return the next required shot for the object in strict order.

        Example:
            get_next_required_shot(obj_id) -> {"env_code": "Clean", ...}
        """
        obj = self._find_object(object_id)
        if not obj:
            return _error("NOT_FOUND", "Object not found.", {"object_id": object_id})

        selected_envs = self.state.get("selected_envs") or []
        if len(selected_envs) != 3:
            return _error(
                "ENV_SELECTION_REQUIRED",
                "Select 3 environments before capturing shots.",
                {"selected_env_codes": selected_envs},
            )

        next_required = _next_required_env(obj["shots"], selected_envs)
        if next_required is None:
            obj["completed"] = True
            self._persist_state()
            return {
                "env_code": None,
                "remaining_needed_for_env": 0,
                "object_completed": True,
            }
        return next_required

    def submit_shot(
        self,
        object_id: str,
        env_code: str,
        image_base64: str,
        accept: bool,
        image_ext: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a shot attempt (accepted or rejected).

        Transport rule:
        - image_base64 is required (base64-encoded bytes of the image).
        - image_path/content_uri are NOT supported in this backend.

        Example:
            submit_shot(obj_id, "Clean", image_b64, True)
        """
        obj = self._find_object(object_id)
        if not obj:
            return _error("NOT_FOUND", "Object not found.", {"object_id": object_id})

        selected_envs = self.state.get("selected_envs") or []
        if len(selected_envs) != 3:
            return _error(
                "ENV_SELECTION_REQUIRED",
                "Select 3 environments before capturing shots.",
                {"selected_env_codes": selected_envs},
            )

        next_required = _next_required_env(obj["shots"], selected_envs)
        if next_required is None:
            return _error("OBJECT_COMPLETE", "Object is already complete.", {"object_id": object_id})
        if env_code not in ALL_ENV_CODES:
            return _error("INVALID_ENV", "env_code is not part of EnvironmentSpec.", {"env_code": env_code})

        if env_code != next_required["env_code"]:
            return _error(
                "WRONG_ENV",
                "Submitted env_code does not match the required env_code.",
                {"required": next_required["env_code"], "submitted": env_code},
            )

        if not image_base64:
            return _error("INVALID_IMAGE", "image_base64 is required.", {})

        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            return _error("INVALID_IMAGE", "image_base64 could not be decoded.", {"error": str(exc)})

        try:
            np, Image = _load_image_libs()
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:  # pylint: disable=broad-except
            return _error("INVALID_IMAGE", "Image data could not be loaded.", {"error": str(exc)})

        # Rejected shots are never persisted (no array, no original, no record).
        if not accept:
            return {
                "accepted": False,
                "deleted_rejected_attempt": True,
                "object_completed": False,
                "next_required": self.get_next_required_shot(object_id),
            }

        sequence = self._next_sequence(obj)
        safe_ext = (image_ext or ".png").lower()
        if not safe_ext.startswith("."):
            safe_ext = f".{safe_ext}"
        if not re.match(r"^\.[a-z0-9]+$", safe_ext):
            return _error("INVALID_IMAGE_EXT", "image_ext must be alphanumeric.", {"image_ext": safe_ext})

        # Filename rule: [Object][EnvironmentCode][Sequence]
        # We use underscores as separators for readability and consistency.
        filename_stem = f"{obj['name']}_{env_code}_{sequence}"
        original_path = self._object_originals_dir(obj["name"]) / f"{filename_stem}{safe_ext}"
        array_path = self._object_arrays_dir(obj["name"]) / f"{filename_stem}.npy"

        # Save original and array. Accepted shots always persist both.
        pil_image.save(original_path)
        np_array = np.asarray(pil_image, dtype=np.uint8)
        np.save(array_path, np_array)

        shot_record = {
            "env_code": env_code,
            "sequence": sequence,
            "original_image_path": str(original_path),
            "array_path": str(array_path),
        }
        obj["shots"].append(shot_record)
        obj["completed"] = _object_completed(obj["shots"], selected_envs)
        self._persist_state()

        if obj["completed"]:
            selected_envs = self.state.get("selected_envs") or []
            if len(selected_envs) == 3:
                training_result = self._build_training_for_object(obj)
                if "error" not in training_result:
                    response = {
                        "accepted": True,
                        "deleted_rejected_attempt": False,
                        "object_completed": obj["completed"],
                        "next_required": None,
                        "training_path": training_result.get("training_path"),
                    }
                    return response

        response = {
            "accepted": True,
            "deleted_rejected_attempt": False,
            "object_completed": obj["completed"],
            "next_required": None if obj["completed"] else self.get_next_required_shot(object_id),
        }
        return response

    def can_select_environments(self) -> Dict[str, Any]:
        """
        Check if environment selection is allowed.
        """
        if any(obj.get("shots") for obj in self.state["objects"]):
            return {"ok": False, "reason": "Environment selection is locked after capturing shots."}
        if self.state["global_state"] != STATE_SELECTING_ENVS:
            self.state["global_state"] = STATE_SELECTING_ENVS
            self._persist_state()
        return {"ok": True, "reason": "Environment selection is available."}

    def add_selected_training_environment(self, env_code: str, confirm: bool) -> Dict[str, Any]:
        """
        Add one environment to the training selection (one-by-one with confirmation).

        Rules:
        - Selection must happen before capturing any shots.
        - Only non-Clean environment codes are allowed.
        - Exactly 3 distinct environments must be chosen.
        - Each selection must be explicitly confirmed by the UI.
        """
        if not confirm:
            return _error("CONFIRM_REQUIRED", "Selection must be confirmed by the user.", {})

        if self.state["global_state"] not in [STATE_SELECTING_ENVS, STATE_COLLECTING_OBJECTS]:
            return _error(
                "STATE_ERROR",
                "Environment selection is not allowed in the current state.",
                {"state": self.state["global_state"]},
            )

        if not self.can_select_environments()["ok"]:
            return _error(
                "NOT_READY",
                "Environment selection is locked after capturing shots.",
                {},
            )

        if env_code not in ALL_ENV_CODES or env_code == "Clean":
            return _error(
                "INVALID_SELECTION",
                "Selection must be a valid non-Clean env code.",
                {"env_code": env_code},
            )

        # Initialize selection list on first pick.
        if self.state["selected_envs"] is None:
            self.state["selected_envs"] = []

        if env_code in self.state["selected_envs"]:
            return _error("INVALID_SELECTION", "Selection contains duplicates.", {"env_code": env_code})

        if len(self.state["selected_envs"]) >= 3:
            return _error("SELECTION_COMPLETE", "Already selected 3 environments.", {})

        self.state["selected_envs"].append(env_code)

        # Persist selection progress for the UI.
        remaining = 3 - len(self.state["selected_envs"])
        if remaining == 0:
            self.state["global_state"] = STATE_TRAINING_READY
            _save_json(
                self.base_dir / "training" / "selected_envs.json",
                {"selected_env_codes": self.state["selected_envs"]},
            )
            for obj in self.state["objects"]:
                if _object_completed(obj["shots"], self.state["selected_envs"]):
                    build_result = self._build_training_for_object(obj)
                    if "error" in build_result:
                        self._persist_state()
                        return build_result
            if self.state["objects"]:
                self.state["global_state"] = STATE_TRAINING_BUILT

        self._persist_state()
        return {"selected_env_codes": self.state["selected_envs"], "remaining_to_select": remaining}

    def set_selected_training_environments(self, env_codes: List[str]) -> Dict[str, Any]:
        """
        Set training selection.

        Rules:
        - Must be set before capturing any shots.
        - Must be exactly 3 distinct, valid, non-Clean env codes.
        - This is a batch helper; the UI should still confirm one-by-one.
        """
        if len(env_codes) != 3:
            return _error("INVALID_SELECTION", "Exactly 3 environments must be selected.", {})

        # Reset selection and add in order to reuse validation logic.
        self.state["selected_envs"] = []
        for env_code in env_codes:
            result = self.add_selected_training_environment(env_code, confirm=True)
            if "error" in result:
                return result
        return self.get_selected_training_environments()

    def get_selected_training_environments(self) -> Dict[str, Any]:
        """Return the current training selection or null."""
        if self.state["selected_envs"] is None:
            return {"selected_env_codes": [], "remaining_to_select": 3}
        remaining = 3 - len(self.state["selected_envs"])
        return {"selected_env_codes": self.state["selected_envs"], "remaining_to_select": remaining}

    def _build_training_for_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        if self.state["selected_envs"] is None:
            return _error("NOT_READY", "Training environments have not been selected.", {})
        if len(self.state["selected_envs"]) != 3:
            return _error("INVALID_SELECTION", "Exactly 3 environments must be selected.", {})
        if not _object_completed(obj["shots"], self.state["selected_envs"]):
            return _error(
                "OBJECT_INCOMPLETE",
                "Object is not complete.",
                {"object_id": obj["object_id"], "object_name": obj["name"]},
            )

        object_name = obj["name"]
        shots = obj["shots"]

        arrays_by_env: Dict[str, List[str]] = {code: [] for code in ALL_ENV_CODES}
        for shot in shots:
            env_code = shot.get("env_code")
            if env_code in arrays_by_env:
                arrays_by_env[env_code].append(shot["array_path"])
        post_training_arrays = _post_training_arrays(shots)

        training_list = {
            "object_name": object_name,
            "shots": [],
            "post_training_arrays": post_training_arrays,
        }

        clean_arrays = arrays_by_env.get("Clean", [])
        if len(clean_arrays) < 2:
            return _error(
                "MISSING_ENV",
                "Object is missing required Clean shots.",
                {"object_name": object_name},
            )
        for clean_path in clean_arrays[:2]:
            training_list["shots"].append({"env_code": "Clean", "array_path": clean_path})

        for code in self.state["selected_envs"]:
            env_arrays = arrays_by_env.get(code, [])
            if not env_arrays:
                return _error(
                    "MISSING_ENV",
                    "Required env_code has no accepted shot.",
                    {"object_name": object_name, "env_code": code},
                )
            training_list["shots"].append({"env_code": code, "array_path": env_arrays[0]})

        training_path = self.base_dir / "training" / f"Training_{object_name}.json"
        _save_json(training_path, training_list)

        feat_dir = self.base_dir / "training" / "feat_export" / object_name
        feat_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = feat_dir / "manifest.json"
        manifest = {
            "object_name": object_name,
            "clean_arrays": arrays_by_env["Clean"],
            "selected_env_arrays": {code: arrays_by_env[code][0] for code in self.state["selected_envs"]},
            "all_arrays_by_env": arrays_by_env,
            "post_training_arrays": post_training_arrays,
            "environment_spec": ENVIRONMENT_SPEC,
            "selected_env_codes": self.state["selected_envs"],
        }
        _save_json(manifest_path, manifest)
        return {"training_path": str(training_path)}

    def create_pending_shot(self, image_bytes: bytes, image_ext: Optional[str]) -> Dict[str, Any]:
        """
        Store a pending shot on disk for Phase 2 confirmation/correction.

        The pending shot is NOT associated with any object yet and does NOT
        consume sequence numbers. It is deleted on commit/cancel.

        Returns:
            {"pending_id": "...", "original_path": "...", "array_path": "...", "image_ext": "jpg"}
        """
        if not image_bytes:
            return _error("INVALID_IMAGE", "image_bytes is required.", {})

        try:
            np, Image = _load_image_libs()
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:  # pylint: disable=broad-except
            return _error("INVALID_IMAGE", "Image data could not be loaded.", {"error": str(exc)})

        safe_ext = (image_ext or ".png").lower()
        if not safe_ext.startswith("."):
            safe_ext = f".{safe_ext}"
        if not re.match(r"^\.[a-z0-9]+$", safe_ext):
            return _error("INVALID_IMAGE_EXT", "image_ext must be alphanumeric.", {"image_ext": safe_ext})

        pending_id = str(uuid.uuid4())
        pending_dir = self.pending_dir / pending_id
        pending_dir.mkdir(parents=True, exist_ok=True)

        original_path = pending_dir / f"original{safe_ext}"
        array_path = pending_dir / "array.npy"
        try:
            pil_image.save(original_path)
            np_array = np.asarray(pil_image, dtype=np.uint8)
            np.save(array_path, np_array)
        except Exception as exc:  # pylint: disable=broad-except
            shutil.rmtree(pending_dir, ignore_errors=True)
            return _error("PENDING_SAVE_FAILED", "Failed to store pending shot.", {"error": str(exc)})

        record = {
            "pending_id": pending_id,
            "original_path": str(original_path),
            "array_path": str(array_path),
            "image_ext": safe_ext.lstrip("."),
        }
        self.pending_shots[pending_id] = record
        return record

    def get_pending_shot(self, pending_id: str) -> Dict[str, Any]:
        """Return the pending record or an error if missing/expired."""
        record = self.pending_shots.get(pending_id)
        if not record:
            return _error("PENDING_NOT_FOUND", "pending_id is missing or expired.", {"pending_id": pending_id})
        return record

    def cancel_pending_shot(self, pending_id: str) -> Dict[str, Any]:
        """
        Cancel a pending shot and delete all temporary files.
        """
        record = self.pending_shots.pop(pending_id, None)
        if not record:
            return _error("PENDING_NOT_FOUND", "pending_id is missing or expired.", {"pending_id": pending_id})
        pending_dir = Path(record["original_path"]).parent
        if pending_dir.exists():
            shutil.rmtree(pending_dir, ignore_errors=True)
        return {"canceled": True}

    def commit_pending_shot(self, pending_id: str, object_id: str, env_code: str) -> Dict[str, Any]:
        """
        Commit a pending shot to the selected object and delete temp artifacts.

        Rules:
        - env_code must be "PostTraining" for Phase 2 commits.
        - sequence numbers increment only on accepted commits.
        """
        if env_code != POST_TRAINING_CODE:
            return _error(
                "INVALID_ENV",
                "Phase 2 commits must use env_code=PostTraining.",
                {"env_code": env_code},
            )

        record = self.pending_shots.get(pending_id)
        if not record:
            return _error("PENDING_NOT_FOUND", "pending_id is missing or expired.", {"pending_id": pending_id})

        obj = self._find_object(object_id)
        if not obj:
            return _error("NOT_FOUND", "Object not found.", {"object_id": object_id})

        sequence = self._next_sequence(obj)
        safe_ext = f".{record['image_ext']}"
        if not re.match(r"^\.[a-z0-9]+$", safe_ext):
            return _error("INVALID_IMAGE_EXT", "image_ext must be alphanumeric.", {"image_ext": safe_ext})

        filename_stem = f"{obj['name']}_{env_code}_{sequence}"
        original_path = self._object_originals_dir(obj["name"]) / f"{filename_stem}{safe_ext}"
        array_path = self._object_arrays_dir(obj["name"]) / f"{filename_stem}.npy"

        # Move pending artifacts into the object folder to make the commit atomic.
        try:
            shutil.move(record["original_path"], original_path)
            shutil.move(record["array_path"], array_path)
            pending_dir = Path(record["original_path"]).parent
            if pending_dir.exists():
                shutil.rmtree(pending_dir, ignore_errors=True)
        except Exception as exc:  # pylint: disable=broad-except
            return _error("COMMIT_FAILED", "Failed to commit pending shot.", {"error": str(exc)})

        shot_record = {
            "env_code": env_code,
            "sequence": sequence,
            "original_image_path": str(original_path),
            "array_path": str(array_path),
        }
        obj["shots"].append(shot_record)
        obj["completed"] = _object_completed(obj["shots"], self.state.get("selected_envs") or [])
        self._persist_state()

        # Update training artifacts to include PostTraining metadata when available.
        self._update_post_training_artifacts(obj)

        # Pending record is no longer valid after commit.
        self.pending_shots.pop(pending_id, None)

        return {
            "committed": True,
            "object_id": obj["object_id"],
            "object_name": obj["name"],
            "sequence": sequence,
        }

    def commit_pending_shot_seed(self, pending_id: str, object_id: str, env_code: str) -> Dict[str, Any]:
        """
        Commit a pending shot as a training seed (Clean/selected envs).
        """
        if env_code not in ALL_ENV_CODES:
            return _error(
                "INVALID_ENV",
                "Seed commits must use a training env_code.",
                {"env_code": env_code},
            )

        record = self.pending_shots.get(pending_id)
        if not record:
            return _error("PENDING_NOT_FOUND", "pending_id is missing or expired.", {"pending_id": pending_id})

        obj = self._find_object(object_id)
        if not obj:
            return _error("NOT_FOUND", "Object not found.", {"object_id": object_id})

        sequence = self._next_sequence(obj)
        safe_ext = f".{record['image_ext']}"
        if not re.match(r"^\.[a-z0-9]+$", safe_ext):
            return _error("INVALID_IMAGE_EXT", "image_ext must be alphanumeric.", {"image_ext": safe_ext})

        filename_stem = f"{obj['name']}_{env_code}_{sequence}"
        original_path = self._object_originals_dir(obj["name"]) / f"{filename_stem}{safe_ext}"
        array_path = self._object_arrays_dir(obj["name"]) / f"{filename_stem}.npy"

        try:
            shutil.move(record["original_path"], original_path)
            shutil.move(record["array_path"], array_path)
            pending_dir = Path(record["original_path"]).parent
            if pending_dir.exists():
                shutil.rmtree(pending_dir, ignore_errors=True)
        except Exception as exc:  # pylint: disable=broad-except
            return _error("COMMIT_FAILED", "Failed to commit pending shot.", {"error": str(exc)})

        shot_record = {
            "env_code": env_code,
            "sequence": sequence,
            "original_image_path": str(original_path),
            "array_path": str(array_path),
        }
        obj["shots"].append(shot_record)
        obj["completed"] = _object_completed(obj["shots"], self.state.get("selected_envs") or [])
        self._persist_state()

        if obj["completed"]:
            training_result = self._build_training_for_object(obj)
            if "error" in training_result:
                return training_result

        self.pending_shots.pop(pending_id, None)
        return {
            "committed": True,
            "object_id": obj["object_id"],
            "object_name": obj["name"],
            "sequence": sequence,
        }

    def _update_post_training_artifacts(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update per-object training artifacts with PostTraining arrays.

        We keep the 5-shot Training_<object>.json intact for FEAT training,
        but add a post_training_arrays field so Phase 2 samples are tracked.
        The FEAT manifest is always updated/created to include PostTraining arrays
        without requiring the object to be complete.
        """
        post_training_arrays = _post_training_arrays(obj["shots"])
        if not post_training_arrays:
            return {"updated": False}

        object_name = obj["name"]
        arrays_by_env: Dict[str, List[str]] = {code: [] for code in ALL_ENV_CODES}
        for shot in obj["shots"]:
            env_code = shot.get("env_code")
            if env_code in arrays_by_env:
                arrays_by_env[env_code].append(shot["array_path"])

        training_path = self.base_dir / "training" / f"Training_{object_name}.json"
        if training_path.exists():
            training_list = _load_json(training_path)
            training_list["post_training_arrays"] = post_training_arrays
        else:
            # Create a minimal training list so PostTraining samples are tracked,
            # while FEAT training can skip incomplete lists safely.
            training_list = {
                "object_name": object_name,
                "shots": [],
                "post_training_arrays": post_training_arrays,
            }
            for clean_path in arrays_by_env.get("Clean", [])[:2]:
                training_list["shots"].append({"env_code": "Clean", "array_path": clean_path})
            for code in self.state.get("selected_envs") or []:
                env_arrays = arrays_by_env.get(code, [])
                if env_arrays:
                    training_list["shots"].append({"env_code": code, "array_path": env_arrays[0]})
        _save_json(training_path, training_list)

        feat_dir = self.base_dir / "training" / "feat_export" / object_name
        feat_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = feat_dir / "manifest.json"
        manifest = {
            "object_name": object_name,
            "clean_arrays": arrays_by_env.get("Clean", []),
            "selected_env_arrays": {
                code: arrays_by_env[code][0]
                for code in (self.state.get("selected_envs") or [])
                if arrays_by_env.get(code)
            },
            "all_arrays_by_env": arrays_by_env,
            "post_training_arrays": post_training_arrays,
            "environment_spec": ENVIRONMENT_SPEC,
            "selected_env_codes": self.state.get("selected_envs") or [],
            "complete": _object_completed(obj["shots"], self.state.get("selected_envs") or []),
        }
        _save_json(manifest_path, manifest)
        return {"updated": True}

    def rebuild_training_for_object(self, object_id: str) -> Dict[str, Any]:
        """
        Rebuild training JSON + FEAT manifest for a single object.

        This is used after Phase 2 post-training sequences where the global
        selected_envs.json already exists and should be reused.
        """
        if self.state["selected_envs"] is None:
            return _error("NOT_READY", "Training environments have not been selected.", {})
        if len(self.state["selected_envs"]) != 3:
            return _error("INVALID_SELECTION", "Exactly 3 environments must be selected.", {})

        obj = self._find_object(object_id)
        if not obj:
            return _error("NOT_FOUND", "Object not found.", {"object_id": object_id})

        if not _object_completed(obj["shots"], self.state["selected_envs"]):
            return _error(
                "OBJECT_INCOMPLETE",
                "Object is not complete.",
                {"object_id": obj["object_id"], "object_name": obj["name"]},
            )

        return self._build_training_for_object(obj)

    def soft_reset_to_baseline(self) -> Dict[str, Any]:
        """
        Remove Phase 2 artifacts and return to the baseline object set.

        - Deletes PostTraining shots for baseline objects.
        - Deletes objects created after inference started.
        - Leaves initial 5-shot training lists intact.
        """
        selected_envs = self.state.get("selected_envs") or []
        if len(selected_envs) != 3:
            return _error("ENV_SELECTION_REQUIRED", "Select 3 environments before soft reset.", {})

        baseline_ids = self.state.get("baseline_object_ids") or []
        if not baseline_ids:
            return _error("BASELINE_MISSING", "Baseline objects are not defined yet.", {})

        baseline_objects = [obj for obj in self.state["objects"] if obj.get("object_id") in baseline_ids]
        if not baseline_objects:
            return _error("BASELINE_MISSING", "No baseline objects available.", {})

        removed_objects: List[str] = []
        removed_post_training: Dict[str, int] = {}

        kept_objects: List[Dict[str, Any]] = []
        for obj in self.state["objects"]:
            name = obj.get("name", "")
            if obj.get("object_id") in baseline_ids:
                kept_objects.append(obj)
                continue
            removed_objects.append(name)
            obj_dir = self._object_dir(name)
            if obj_dir.exists():
                shutil.rmtree(obj_dir, ignore_errors=True)
            training_path = self.base_dir / "training" / f"Training_{name}.json"
            if training_path.exists():
                training_path.unlink()
            feat_dir = self.base_dir / "training" / "feat_export" / name
            if feat_dir.exists():
                shutil.rmtree(feat_dir, ignore_errors=True)

        for obj in kept_objects:
            removed = []
            kept = []
            for shot in obj.get("shots", []):
                if shot.get("env_code") == POST_TRAINING_CODE:
                    removed.append(shot)
                else:
                    kept.append(shot)

            if removed:
                for shot in removed:
                    original = Path(shot.get("original_image_path", ""))
                    array = Path(shot.get("array_path", ""))
                    if original.exists():
                        original.unlink()
                    if array.exists():
                        array.unlink()

            obj["shots"] = kept
            obj["completed"] = _object_completed(obj["shots"], selected_envs)
            removed_post_training[obj.get("name", "")] = len(removed)

            if not obj["completed"]:
                return _error(
                    "OBJECT_INCOMPLETE",
                    "Baseline object is incomplete.",
                    {"object_id": obj.get("object_id"), "object_name": obj.get("name")},
                )
            rebuild = self._build_training_for_object(obj)
            if "error" in rebuild:
                return rebuild

        # Reset pending inference artifacts to stop the session cleanly.
        if self.pending_dir.exists():
            shutil.rmtree(self.pending_dir, ignore_errors=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.pending_shots = {}

        # Update class map to contain only baseline objects (if present).
        class_map_path = self.base_dir / "training" / "class_map.json"
        if class_map_path.exists():
            try:
                payload = _load_json(class_map_path)
                name_to_id = payload.get("name_to_id", {}) or {}
                next_id = int(payload.get("next_id", 0))
                baseline_names = [obj.get("name", "") for obj in kept_objects if obj.get("name")]
                for name in baseline_names:
                    if name not in name_to_id:
                        name_to_id[name] = next_id
                        next_id += 1
                name_to_id = {name: name_to_id[name] for name in baseline_names}
                payload.update(
                    {
                        "name_to_id": name_to_id,
                        "next_id": next_id,
                        "active_names": baseline_names,
                    }
                )
                _save_json(class_map_path, payload)
            except Exception:  # pylint: disable=broad-except
                class_map_path.unlink(missing_ok=True)

        cache_path = self.base_dir / "training" / "og_feat_support_cache.pt"
        if cache_path.exists():
            cache_path.unlink()

        self.state["objects"] = kept_objects
        self.state["baseline_object_ids"] = [obj.get("object_id") for obj in kept_objects if obj.get("object_id")]
        self.state["global_state"] = STATE_TRAINING_BUILT
        self._persist_state()

        return {
            "soft_reset": True,
            "baseline_objects": len(kept_objects),
            "removed_objects": removed_objects,
            "removed_post_training": removed_post_training,
        }

    def build_training_sets(self) -> Dict[str, Any]:
        """
        Build training sets and FEAT export manifests.
        """
        if self.state["selected_envs"] is None:
            return _error("NOT_READY", "Training environments have not been selected.", {})
        if len(self.state["selected_envs"]) != 3:
            return _error("INVALID_SELECTION", "Exactly 3 environments must be selected.", {})
        if self.state["global_state"] not in [STATE_TRAINING_READY, STATE_TRAINING_BUILT]:
            return _error("NOT_READY", "Backend is not in TRAINING_READY state.", {})

        training_files: List[str] = []
        for obj in self.state["objects"]:
            build_result = self._build_training_for_object(obj)
            if "error" in build_result:
                return build_result
            training_files.append(build_result["training_path"])

        self.state["global_state"] = STATE_TRAINING_BUILT
        self._persist_state()
        return {"built": True, "training_files": training_files}

    def reset_all(self) -> Dict[str, Any]:
        """
        Delete all objects and reset backend state to defaults.
        """
        raw_dir = self.base_dir / "raw"
        training_dir = self.base_dir / "training"
        pending_dir = self.pending_dir
        config_dir = self.base_dir / "config"
        thresholds_path = config_dir / "thresholds.json"
        inference_mode_path = config_dir / "inference_mode.json"
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        if training_dir.exists():
            shutil.rmtree(training_dir)
        if pending_dir.exists():
            shutil.rmtree(pending_dir)
        if thresholds_path.exists():
            thresholds_path.unlink()
        if inference_mode_path.exists():
            inference_mode_path.unlink()
        if config_dir.exists():
            try:
                config_dir.rmdir()
            except OSError:
                pass
        self.pending_shots = {}
        self._ensure_base_layout()
        self.state = {
            "version": 1,
            "global_state": STATE_SELECTING_ENVS,
            "next_instance_id": 1,
            "objects": [],
            "selected_envs": [],
            "inference_started": False,
            "baseline_object_ids": [],
        }
        self._persist_state()
        return {"ok": True}

    def self_check(self) -> Dict[str, Any]:
        """
        Validate base_dir layout and state.json consistency without modifying state.

        Returns:
            {"ok": bool, "issues": [...], "state": "<global_state>"}
        """
        issues: List[Dict[str, Any]] = []

        def add_issue(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
            issues.append({"code": code, "message": message, "details": details or {}})

        # Base folders required by the storage contract.
        required_dirs = [
            self.base_dir / "raw",
            self.base_dir / "training",
            self.base_dir / "training" / "feat_export",
        ]
        for path in required_dirs:
            if not path.exists():
                add_issue("MISSING_DIR", "Required directory is missing.", {"path": str(path)})

        # State.json required fields.
        for field in [
            "version",
            "global_state",
            "next_instance_id",
            "objects",
            "selected_envs",
            "inference_started",
            "baseline_object_ids",
        ]:
            if field not in self.state:
                add_issue("MISSING_STATE_FIELD", "Required state field is missing.", {"field": field})

        # Object folders and shot files.
        for obj in self.state.get("objects", []):
            name = obj.get("name")
            if not name:
                add_issue("INVALID_OBJECT", "Object has no name.", {"object_id": obj.get("object_id")})
                continue
            originals_dir = self._object_originals_dir(name)
            arrays_dir = self._object_arrays_dir(name)
            if not originals_dir.exists():
                add_issue("MISSING_DIR", "Originals directory is missing.", {"path": str(originals_dir)})
            if not arrays_dir.exists():
                add_issue("MISSING_DIR", "Arrays directory is missing.", {"path": str(arrays_dir)})

            for shot in obj.get("shots", []):
                original_path = Path(shot.get("original_image_path", ""))
                array_path = Path(shot.get("array_path", ""))
                if not original_path.exists():
                    add_issue("MISSING_FILE", "Original image is missing.", {"path": str(original_path)})
                if not array_path.exists():
                    add_issue("MISSING_FILE", "Array file is missing.", {"path": str(array_path)})

        return {"ok": len(issues) == 0, "issues": issues, "state": self.state.get("global_state")}

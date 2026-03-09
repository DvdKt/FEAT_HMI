from __future__ import annotations

import base64
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

ROOT_DIR = Path(__file__).resolve().parent
PY_BACKEND_DIR = ROOT_DIR / "app" / "src" / "main" / "python"
if str(PY_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(PY_BACKEND_DIR))

import backend as py_backend  # noqa: E402  # isort: skip
from constants import POST_TRAINING_CODE  # noqa: E402  # isort: skip
from core import Backend  # noqa: E402  # isort: skip
from feat_inference import (  # noqa: E402  # isort: skip
    ThresholdConfig,
    refresh_support_cache,
    run_feat_inference,
)


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    data_dir: Path
    api_key: str
    enforce_api_key: bool
    conf_threshold: float
    margin_threshold: float
    auto_train_feat: bool


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _load_settings() -> Settings:
    env_path = ROOT_DIR / "backend" / ".env"
    load_dotenv(env_path, override=False)

    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port_raw = os.getenv("BACKEND_PORT", "8000")
    data_dir_raw = os.getenv("DATA_DIR", str(Path.home() / "WerkzeugerkennungData"))
    api_key = os.getenv("API_KEY", "").strip()
    require_api_key_raw = os.getenv("REQUIRE_API_KEY", "false")
    conf_raw = os.getenv("T_CONF", "0.8")
    margin_raw = os.getenv("T_MARGIN", "0.1")
    auto_train_raw = os.getenv("AUTO_TRAIN_FEAT", "true")

    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ValueError(f"BACKEND_PORT must be an integer, got {port_raw!r}") from exc

    try:
        require_api_key = _parse_bool(require_api_key_raw)
    except ValueError as exc:
        raise ValueError(f"REQUIRE_API_KEY must be true/false, got {require_api_key_raw!r}") from exc

    try:
        conf_threshold = float(conf_raw)
        margin_threshold = float(margin_raw)
    except ValueError as exc:
        raise ValueError("T_CONF and T_MARGIN must be floats.") from exc
    try:
        auto_train_feat = _parse_bool(auto_train_raw)
    except ValueError as exc:
        raise ValueError(f"AUTO_TRAIN_FEAT must be true/false, got {auto_train_raw!r}") from exc

    enforce_api_key = bool(api_key) or require_api_key
    if require_api_key and not api_key:
        raise ValueError("REQUIRE_API_KEY is true but API_KEY is empty.")

    return Settings(
        host=host,
        port=port,
        data_dir=_expand_path(data_dir_raw),
        api_key=api_key,
        enforce_api_key=enforce_api_key,
        conf_threshold=conf_threshold,
        margin_threshold=margin_threshold,
        auto_train_feat=auto_train_feat,
    )


SETTINGS = _load_settings()
app = FastAPI(title="Werkzeugerkennung Backend", version="1.0")

# Phase 2 pending inference metadata is stored in-memory only. The authoritative
# pending shot artifacts live on disk under base_dir/pending/.
_PENDING_INFERENCE: Dict[str, Dict[str, Any]] = {}


def _thresholds_path() -> Path:
    return SETTINGS.data_dir / "config" / "thresholds.json"


def _inference_mode_path() -> Path:
    return SETTINGS.data_dir / "config" / "inference_mode.json"


def _load_inference_mode() -> Dict[str, Any]:
    """
    Load persisted inference mode or return empty if missing/invalid.
    """
    path = _inference_mode_path()
    if not path.exists():
        return {"mode": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pylint: disable=broad-except
        return {"mode": None}
    mode = str(payload.get("mode", "")).strip().lower()
    if mode not in {"semi-automatic", "full-automatic"}:
        return {"mode": None}
    return {"mode": mode}


def _save_inference_mode(mode: str) -> None:
    path = _inference_mode_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": mode}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_threshold_config() -> Dict[str, Any]:
    """
    Load persisted threshold configuration or return defaults if missing.
    """
    path = _thresholds_path()
    if not path.exists():
        return {
            "conf_threshold": SETTINGS.conf_threshold,
            "margin_threshold": SETTINGS.margin_threshold,
            "locked": False,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pylint: disable=broad-except
        return {
            "conf_threshold": SETTINGS.conf_threshold,
            "margin_threshold": SETTINGS.margin_threshold,
            "locked": False,
        }
    try:
        conf = float(payload.get("conf_threshold", SETTINGS.conf_threshold))
        margin = float(payload.get("margin_threshold", SETTINGS.margin_threshold))
    except (TypeError, ValueError):
        conf = SETTINGS.conf_threshold
        margin = SETTINGS.margin_threshold
    locked = bool(payload.get("locked", False))
    return {"conf_threshold": conf, "margin_threshold": margin, "locked": locked}


def _save_threshold_config(conf_threshold: float, margin_threshold: float, locked: bool) -> None:
    path = _thresholds_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "conf_threshold": conf_threshold,
        "margin_threshold": margin_threshold,
        "locked": locked,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_thresholds(conf_threshold: float, margin_threshold: float) -> Optional[JSONResponse]:
    if not (0.0 <= conf_threshold <= 1.0):
        return _error_response(
            "INVALID_THRESHOLD",
            "conf_threshold must be between 0.0 and 1.0.",
            {"conf_threshold": conf_threshold},
        )
    if not (0.0 <= margin_threshold <= 1.0):
        return _error_response(
            "INVALID_THRESHOLD",
            "margin_threshold must be between 0.0 and 1.0.",
            {"margin_threshold": margin_threshold},
        )
    return None


def _validate_inference_mode(mode: str) -> Optional[JSONResponse]:
    if mode not in {"semi-automatic", "full-automatic"}:
        return _error_response(
            "INVALID_INFERENCE_MODE",
            "mode must be 'semi-automatic' or 'full-automatic'.",
            {"mode": mode},
        )
    return None


def _unlock_threshold_config() -> None:
    current = _load_threshold_config()
    _save_threshold_config(current["conf_threshold"], current["margin_threshold"], False)


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not SETTINGS.enforce_api_key:
        return
    if not x_api_key or x_api_key != SETTINGS.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key.")


def _error_response(
    code: str, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 400
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details or {}}},
    )


def _unwrap_result(result: Any) -> Any:
    if isinstance(result, dict) and "error" in result:
        error = result["error"]
        message = error.get("message", "Backend error")
        code = error.get("code")
        detail = f"{code}: {message}" if code else message
        raise HTTPException(status_code=400, detail=detail)
    return result


def _map_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "object_id": obj.get("object_id", ""),
        "object_name": obj.get("name") or obj.get("object_name") or "",
        "instance_id": obj.get("instance_id", ""),
        "completed": obj.get("completed", False),
    }


def _get_backend() -> Optional[Backend]:
    backend = getattr(py_backend, "_BACKEND", None)
    return backend


def _ensure_feat_model() -> Optional[JSONResponse]:
    """
    Ensure the OG_FEAT model artifact exists before inference.

    This auto-trains when the model is missing or stale so inference stays in sync.
    """
    model_path = SETTINGS.data_dir / "training" / "og_feat_model.pt"
    training_dir = SETTINGS.data_dir / "training"
    training_lists = list(training_dir.glob("Training_*.json"))
    latest_training_mtime = max((path.stat().st_mtime for path in training_lists), default=0.0)

    if model_path.exists() and model_path.stat().st_mtime >= latest_training_mtime:
        return None

    if not SETTINGS.auto_train_feat:
        return _error_response(
            "FEAT_MODEL_MISSING",
            "OG_FEAT model is missing or stale and auto-training is disabled.",
            {"error": f"OG_FEAT model not found at {model_path}"},
        )

    if not training_lists:
        build_result = _result_or_error(py_backend.build_training_sets())
        if isinstance(build_result, JSONResponse):
            return build_result

    train_result = _result_or_error(py_backend.train_feat_model())
    if isinstance(train_result, JSONResponse):
        return train_result

    if not model_path.exists():
        return _error_response(
            "MODEL_LOAD_FAILED",
            "Failed to load OG_FEAT model.",
            {"error": f"OG_FEAT model not found at {model_path}"},
        )
    return None


def _refresh_support_cache() -> None:
    try:
        refresh_support_cache(SETTINGS.data_dir)
    except Exception:  # pylint: disable=broad-except
        pass


_REFRESH_LOCK = threading.Lock()
_REFRESH_IN_FLIGHT = False


def _refresh_support_cache_async() -> None:
    """
    Refresh the support cache in a background thread to avoid request timeouts.
    """
    def _run() -> None:
        global _REFRESH_IN_FLIGHT  # noqa: PLW0603 - guarded by _REFRESH_LOCK
        try:
            _refresh_support_cache()
        finally:
            with _REFRESH_LOCK:
                _REFRESH_IN_FLIGHT = False

    global _REFRESH_IN_FLIGHT  # noqa: PLW0603 - guarded by _REFRESH_LOCK
    with _REFRESH_LOCK:
        if _REFRESH_IN_FLIGHT:
            return
        _REFRESH_IN_FLIGHT = True

    thread = threading.Thread(
        target=_run,
        name="support-cache-refresh",
        daemon=True,
    )
    thread.start()


def _result_or_error(result: Any) -> Any:
    """
    Return JSON error responses in the standard shape if the backend reported an error.
    """
    if isinstance(result, dict) and "error" in result:
        error = result["error"]
        return _error_response(
            error.get("code", "BACKEND_ERROR"),
            error.get("message", "Backend error"),
            error.get("details"),
        )
    return result


@app.on_event("startup")
def _startup() -> None:
    SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
    init = py_backend.initialize_backend(str(SETTINGS.data_dir))
    _unwrap_result(init)


@app.get("/health")
def health() -> Dict[str, Any]:
    threshold_config = _load_threshold_config()
    mode_config = _load_inference_mode()
    check = py_backend.self_check()
    if isinstance(check, dict) and "error" in check:
        error = check["error"]
        return {"status": "error", "detail": error.get("message", "Unknown error")}

    backend_state = check.get("state") if isinstance(check, dict) else None
    version = None
    backend_instance = getattr(py_backend, "_BACKEND", None)
    if backend_instance is not None:
        version = backend_instance.state.get("version")

    return {
        "status": "ok" if check.get("ok") else "degraded",
        "state": backend_state,
        "version": version,
        "issues": check.get("issues", []),
        "conf_threshold": threshold_config["conf_threshold"],
        "margin_threshold": threshold_config["margin_threshold"],
        "thresholds_locked": threshold_config["locked"],
        "inference_mode": mode_config["mode"],
    }


@app.get("/config")
def get_config() -> Dict[str, Any]:
    """
    Return backend configuration relevant to inference debugging.

    Example response:
        {"conf_threshold": 0.8, "margin_threshold": 0.1, "data_dir": "..."}
    """
    threshold_config = _load_threshold_config()
    mode_config = _load_inference_mode()
    return {
        "conf_threshold": threshold_config["conf_threshold"],
        "margin_threshold": threshold_config["margin_threshold"],
        "thresholds_locked": threshold_config["locked"],
        "inference_mode": mode_config["mode"],
        "data_dir": str(SETTINGS.data_dir),
    }


@app.get("/thresholds")
def get_thresholds() -> Dict[str, Any]:
    """
    Return the current threshold configuration and lock status.

    Example response:
        {"conf_threshold": 0.8, "margin_threshold": 0.1, "locked": false}
    """
    return _load_threshold_config()


@app.post("/thresholds", dependencies=[Depends(_require_api_key)])
def set_thresholds(payload: Dict[str, Any] = Body(...)) -> Any:
    """
    Set thresholds once and lock them.

    Input:
        {"conf_threshold": 0.8, "margin_threshold": 0.1}
    """
    current = _load_threshold_config()
    if current.get("locked"):
        return _error_response(
            "THRESHOLDS_LOCKED",
            "Thresholds are already locked and cannot be changed.",
            {},
        )

    conf_raw = payload.get("conf_threshold")
    margin_raw = payload.get("margin_threshold")
    if conf_raw is None or margin_raw is None:
        return _error_response(
            "INVALID_THRESHOLD",
            "conf_threshold and margin_threshold are required.",
            {"payload": payload},
        )
    try:
        conf = float(conf_raw)
        margin = float(margin_raw)
    except (TypeError, ValueError):
        return _error_response(
            "INVALID_THRESHOLD",
            "conf_threshold and margin_threshold must be numbers.",
            {"payload": payload},
        )

    error = _validate_thresholds(conf, margin)
    if error is not None:
        return error

    _save_threshold_config(conf, margin, True)
    return {"conf_threshold": conf, "margin_threshold": margin, "locked": True}


@app.get("/inference-mode")
def get_inference_mode() -> Dict[str, Any]:
    """
    Return the current inference mode (semi-automatic or full-automatic).
    """
    return _load_inference_mode()


@app.post("/inference-mode", dependencies=[Depends(_require_api_key)])
def set_inference_mode(payload: Dict[str, Any] = Body(...)) -> Any:
    """
    Set the inference mode used for Phase 2.

    Input:
        {"mode": "semi-automatic"} or {"mode": "full-automatic"}
    """
    mode_raw = payload.get("mode")
    if not isinstance(mode_raw, str) or not mode_raw.strip():
        return _error_response(
            "INVALID_INFERENCE_MODE",
            "mode is required.",
            {"mode": mode_raw},
        )
    mode = mode_raw.strip().lower()
    error = _validate_inference_mode(mode)
    if error is not None:
        return error
    _save_inference_mode(mode)
    return {"mode": mode}


def _soft_reset_status() -> Any:
    status = py_backend.get_soft_reset_status()
    if isinstance(status, dict) and "error" in status:
        error = status["error"]
        return _error_response(
            error.get("code", "BACKEND_ERROR"),
            error.get("message", "Backend error"),
            error.get("details"),
        )

    threshold_config = _load_threshold_config()
    status.update(
        {
            "thresholds_locked": threshold_config.get("locked", False),
        }
    )

    missing = []
    if not status.get("envs_selected"):
        missing.append("envs_selected")
    if not status.get("objects_created"):
        missing.append("objects_created")
    if not status.get("training_lists_complete"):
        missing.append("training_lists_complete")
    if not status.get("inference_started"):
        missing.append("inference_started")
    if not status.get("thresholds_locked"):
        missing.append("thresholds_locked")

    status["missing"] = missing
    status["can_soft_reset"] = len(missing) == 0
    return status


@app.get("/soft-reset/status")
def get_soft_reset_status() -> Any:
    """
    Return whether a soft reset is allowed and which requirements are missing.
    """
    return _soft_reset_status()


@app.post("/soft-reset", dependencies=[Depends(_require_api_key)])
def soft_reset() -> Any:
    """
    Revert to baseline objects and clear Phase 2 artifacts.
    """
    status = _soft_reset_status()
    if isinstance(status, JSONResponse):
        return status
    if not status.get("can_soft_reset"):
        return _error_response(
            "SOFT_RESET_NOT_READY",
            "Soft reset requirements are not met.",
            {"missing": status.get("missing", [])},
        )

    result = py_backend.soft_reset_to_baseline()
    if isinstance(result, dict) and "error" in result:
        error = result["error"]
        return _error_response(
            error.get("code", "BACKEND_ERROR"),
            error.get("message", "Soft reset failed."),
            error.get("details"),
        )

    _unlock_threshold_config()
    _PENDING_INFERENCE.clear()
    _refresh_support_cache_async()
    return result


@app.get("/env-spec")
def get_env_spec() -> Any:
    return _result_or_error(py_backend.get_env_spec())


@app.get("/objects")
def list_objects() -> Any:
    result = _result_or_error(py_backend.list_objects())
    if isinstance(result, JSONResponse):
        return result
    return [_map_object(obj) for obj in result]


@app.post("/objects", dependencies=[Depends(_require_api_key)])
def create_object(payload: Dict[str, Any] = Body(...)) -> Any:
    object_name = payload.get("object_name") or payload.get("name") or ""
    overwrite = bool(payload.get("overwrite", False))
    result = _result_or_error(py_backend.create_object(object_name, overwrite=overwrite))
    if isinstance(result, JSONResponse):
        return result
    return _map_object(result)


@app.get("/objects/{object_id}/next-shot")
def get_next_shot(object_id: str) -> Any:
    return _result_or_error(py_backend.get_next_required_shot(object_id))


@app.post("/objects/{object_id}/shots", dependencies=[Depends(_require_api_key)])
async def submit_shot(
    object_id: str,
    env_code: str = Form(...),
    accept: str = Form(...),
    image_file: UploadFile = File(...),
) -> Any:
    try:
        accept_bool = _parse_bool(accept)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_bytes = await image_file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    image_ext = Path(image_file.filename or "").suffix.lstrip(".") or "jpg"
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    result = _result_or_error(
        py_backend.submit_shot(
            object_id=object_id,
            env_code=env_code,
            image_base64=image_b64,
            accept=accept_bool,
            image_ext=image_ext,
        )
    )
    if isinstance(result, JSONResponse):
        return result

    # Support cache refresh is deferred to the first inference to avoid timeouts during training.

    status = "accepted" if result.get("accepted") else "rejected"
    return {
        "status": status,
        "next_required_shot": result.get("next_required"),
        "object_completed": result.get("object_completed", False),
    }


@app.get("/training/selected-envs")
def get_selected_training_envs() -> Any:
    return _result_or_error(py_backend.get_selected_training_environments())


@app.post("/training/selected-envs/confirm", dependencies=[Depends(_require_api_key)])
def confirm_training_env(payload: Dict[str, Any] = Body(...)) -> Any:
    env_code = payload.get("env_code") or ""
    confirm_raw = payload.get("confirm", True)
    if isinstance(confirm_raw, str):
        try:
            confirm = _parse_bool(confirm_raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        confirm = bool(confirm_raw)
    result = _result_or_error(py_backend.select_training_environment(env_code, confirm))
    return result


@app.post("/training/selected-envs", dependencies=[Depends(_require_api_key)])
def set_selected_training_envs(payload: Dict[str, Any] = Body(...)) -> Any:
    env_codes = payload.get("selected_env_codes")
    if not isinstance(env_codes, list):
        return _error_response("INVALID_SELECTION", "selected_env_codes must be a list.", {})
    result = _result_or_error(py_backend.set_selected_training_environments(env_codes))
    return result


@app.post("/training/build", dependencies=[Depends(_require_api_key)])
def build_training_sets() -> Any:
    result = _result_or_error(py_backend.build_training_sets_for_ui())
    return result


@app.get("/training/lists")
def list_training_sets() -> Any:
    return _result_or_error(py_backend.list_training_sets())


@app.post("/training/feat", dependencies=[Depends(_require_api_key)])
def train_feat_model() -> Any:
    result = _result_or_error(py_backend.train_feat_model())
    return result


@app.post("/reset", dependencies=[Depends(_require_api_key)])
def reset_all() -> Any:
    result = _result_or_error(py_backend.reset_all())
    if isinstance(result, JSONResponse):
        return result
    return {"ok": bool(result.get("ok"))}


@app.post("/inference", dependencies=[Depends(_require_api_key)])
async def run_inference(image_file: UploadFile = File(...)) -> Any:
    """
    Run FEAT inference on a single image and create a pending shot.

    Input (multipart):
        image_file: image bytes (jpg/png)

    Example response:
    {
      "pending_id": "...",
      "predicted": {
        "object_id": "... or null",
        "object_name": "... or null",
        "predicted_label": "... or null",
        "probs_topk": [{"object_id":"...","object_name":"...","prob":0.73}, ...],
        "max_prob": 0.73,
        "second_prob": 0.18,
        "margin": 0.55,
        "passed_conf": true,
        "passed_margin": true,
        "accepted_by_threshold": true,
        "is_unknown": false,
        "unknown_prob": 0.12,
        "best_known_prob": 0.73,
        "runner_up_label": "...",
        "runner_up_prob": 0.18,
        "confidence": 0.73
      },
      "next_action": "CONFIRM_PREDICTION"
    }
    """
    backend = _get_backend()
    if backend is None:
        return _error_response("NOT_INITIALIZED", "Backend is not initialized.", {})

    mode_config = _load_inference_mode()
    if not mode_config.get("mode"):
        return _error_response(
            "INFERENCE_MODE_REQUIRED",
            "Select Semi-Automatic or Full-Automatic before inference.",
            {},
        )

    started = py_backend.mark_inference_started()
    if isinstance(started, dict) and "error" in started:
        return _error_response(
            started["error"].get("code", "BACKEND_ERROR"),
            started["error"].get("message", "Failed to start inference."),
            started["error"].get("details"),
        )

    auto_train_error = _ensure_feat_model()
    if auto_train_error is not None:
        return auto_train_error

    image_bytes = await image_file.read()
    if not image_bytes:
        return _error_response("INVALID_IMAGE", "Empty image upload.", {})

    image_ext = Path(image_file.filename or "").suffix.lstrip(".") or "jpg"
    pending = backend.create_pending_shot(image_bytes, image_ext)
    if "error" in pending:
        return _error_response(
            pending["error"]["code"],
            pending["error"]["message"],
            pending["error"].get("details"),
        )

    pending_id = pending["pending_id"]
    threshold_config = _load_threshold_config()
    thresholds = ThresholdConfig(
        conf_threshold=threshold_config["conf_threshold"],
        margin_threshold=threshold_config["margin_threshold"],
    )
    inference = run_feat_inference(SETTINGS.data_dir, Path(pending["array_path"]), thresholds)
    if "error" in inference:
        backend.cancel_pending_shot(pending_id)
        return _error_response(
            inference["error"]["code"],
            inference["error"]["message"],
            inference["error"].get("details"),
        )

    objects = backend.state.get("objects", [])
    name_to_id = {obj["name"]: obj["object_id"] for obj in objects}
    probs_topk = []
    for item in inference["probs_topk"]:
        name = item["object_name"]
        probs_topk.append(
            {
                "object_id": name_to_id.get(name),
                "object_name": name,
                "class_id": item.get("class_id"),
                "prob": item["prob"],
            }
        )

    accepted = inference["accepted_by_threshold"]
    top1 = probs_topk[0] if probs_topk else {"object_id": None, "object_name": None}

    if accepted and not top1.get("object_id"):
        backend.cancel_pending_shot(pending_id)
        return _error_response(
            "PREDICTION_MISSING",
            "Prediction passed thresholds but object mapping is missing.",
            {"object_name": top1.get("object_name")},
        )

    predicted = {
        "object_id": top1.get("object_id") if accepted else None,
        "object_name": top1.get("object_name") if accepted else None,
        "class_id": top1.get("class_id") if accepted else None,
        "probs_topk": probs_topk,
        "max_prob": inference["max_prob"],
        "second_prob": inference["second_prob"],
        "margin": inference["margin"],
        "passed_conf": inference["passed_conf"],
        "passed_margin": inference["passed_margin"],
        "accepted_by_threshold": accepted,
        "is_unknown": inference["is_unknown"],
        "predicted_label": inference.get("predicted_label"),
        "predicted_class_id": inference.get("predicted_class_id"),
        "unknown_prob": inference.get("unknown_prob"),
        "best_known_prob": inference.get("best_known_prob"),
        "runner_up_label": inference.get("runner_up_label"),
        "runner_up_class_id": inference.get("runner_up_class_id"),
        "runner_up_prob": inference.get("runner_up_prob"),
        "confidence": inference.get("best_known_prob"),
        "feat_confidence": inference.get("feat_confidence"),
    }
    if "debug" in inference:
        predicted["debug"] = inference["debug"]

    _PENDING_INFERENCE[pending_id] = {
        "predicted_object_id": top1.get("object_id"),
        "predicted_object_name": top1.get("object_name"),
        "accepted_by_threshold": accepted,
    }

    next_action = "CONFIRM_PREDICTION" if accepted else "ASK_NEW_OR_EXISTING"
    return {"pending_id": pending_id, "predicted": predicted, "next_action": next_action}


@app.post("/confirm_prediction", dependencies=[Depends(_require_api_key)])
def confirm_prediction(payload: Dict[str, Any] = Body(...)) -> Any:
    """
    Confirm or reject the model prediction for a pending shot.

    Input:
        {"pending_id": "...", "user_confirms": true}
    """
    backend = _get_backend()
    if backend is None:
        return _error_response("NOT_INITIALIZED", "Backend is not initialized.", {})

    pending_id = payload.get("pending_id") or ""
    user_confirms = bool(payload.get("user_confirms", False))
    pending_check = backend.get_pending_shot(pending_id)
    if "error" in pending_check:
        return _error_response(
            pending_check["error"]["code"],
            pending_check["error"]["message"],
            pending_check["error"].get("details"),
        )

    if not user_confirms:
        return {"needs_correction": True, "allow_new_object": True}

    predicted = _PENDING_INFERENCE.get(pending_id)
    if not predicted or not predicted.get("predicted_object_id"):
        return _error_response(
            "PREDICTION_MISSING",
            "Prediction is missing for pending_id.",
            {"pending_id": pending_id},
        )
    if not predicted.get("accepted_by_threshold"):
        return _error_response(
            "PREDICTION_NOT_ACCEPTED",
            "Prediction did not pass thresholds.",
            {"pending_id": pending_id},
        )

    commit = backend.commit_pending_shot(
        pending_id=pending_id,
        object_id=predicted["predicted_object_id"],
        env_code=POST_TRAINING_CODE,
    )
    if "error" in commit:
        return _error_response(
            commit["error"]["code"],
            commit["error"]["message"],
            commit["error"].get("details"),
        )

    _PENDING_INFERENCE.pop(pending_id, None)
    _refresh_support_cache_async()
    return {"committed": True, "object_id": commit["object_id"], "sequence": commit["sequence"]}


@app.post("/submit_correction", dependencies=[Depends(_require_api_key)])
def submit_correction(payload: Dict[str, Any] = Body(...)) -> Any:
    """
    Commit a pending shot to a user-selected object after correction.

    Input:
        {"pending_id": "...", "object_id": "...", "user_confirms": true}
    """
    backend = _get_backend()
    if backend is None:
        return _error_response("NOT_INITIALIZED", "Backend is not initialized.", {})

    pending_id = payload.get("pending_id") or ""
    object_id = payload.get("object_id") or ""
    user_confirms = bool(payload.get("user_confirms", False))

    pending_check = backend.get_pending_shot(pending_id)
    if "error" in pending_check:
        return _error_response(
            pending_check["error"]["code"],
            pending_check["error"]["message"],
            pending_check["error"].get("details"),
        )

    if not user_confirms:
        return {"needs_correction": True, "allow_new_object": False}

    commit = backend.commit_pending_shot(
        pending_id=pending_id,
        object_id=object_id,
        env_code=POST_TRAINING_CODE,
    )
    if "error" in commit:
        return _error_response(
            commit["error"]["code"],
            commit["error"]["message"],
            commit["error"].get("details"),
        )

    _PENDING_INFERENCE.pop(pending_id, None)
    _refresh_support_cache_async()
    return {"committed": True, "object_id": commit["object_id"], "sequence": commit["sequence"]}


@app.post("/unknown_decision", dependencies=[Depends(_require_api_key)])
def unknown_decision(payload: Dict[str, Any] = Body(...)) -> Any:
    """
    Handle the UNKNOWN decision branch.

    Input:
        {"pending_id": "...", "is_new": true}
    """
    backend = _get_backend()
    if backend is None:
        return _error_response("NOT_INITIALIZED", "Backend is not initialized.", {})

    pending_id = payload.get("pending_id") or ""
    is_new = bool(payload.get("is_new", False))
    pending_check = backend.get_pending_shot(pending_id)
    if "error" in pending_check:
        return _error_response(
            pending_check["error"]["code"],
            pending_check["error"]["message"],
            pending_check["error"].get("details"),
        )

    if not is_new:
        return {"needs_correction": True}

    return {"needs_object_name": True}


@app.post("/create_object_from_pending", dependencies=[Depends(_require_api_key)])
def create_object_from_pending(payload: Dict[str, Any] = Body(...)) -> Any:
    """
    Create a new object and commit the pending shot as a training seed.

    Input:
        {"pending_id": "...", "object_name": "Hammer"}
    """
    backend = _get_backend()
    if backend is None:
        return _error_response("NOT_INITIALIZED", "Backend is not initialized.", {})

    pending_id = payload.get("pending_id") or ""
    object_name = payload.get("object_name") or ""
    pending_check = backend.get_pending_shot(pending_id)
    if "error" in pending_check:
        return _error_response(
            pending_check["error"]["code"],
            pending_check["error"]["message"],
            pending_check["error"].get("details"),
        )

    created = py_backend.create_object(object_name, overwrite=False)
    if isinstance(created, dict) and "error" in created:
        return _error_response(
            created["error"]["code"],
            created["error"]["message"],
            created["error"].get("details"),
        )

    commit = backend.commit_pending_shot_seed(
        pending_id=pending_id,
        object_id=created["object_id"],
        env_code="Clean",
    )
    if "error" in commit:
        return _error_response(
            commit["error"]["code"],
            commit["error"]["message"],
            commit["error"].get("details"),
        )

    _PENDING_INFERENCE.pop(pending_id, None)
    _refresh_support_cache_async()
    return {
        "created": True,
        "object_id": created["object_id"],
        "object_name": created["name"],
        "committed": True,
        "ask_sequence": True,
    }


@app.post("/start_post_training_sequence", dependencies=[Depends(_require_api_key)])
def start_post_training_sequence(payload: Dict[str, Any] = Body(...)) -> Any:
    """
    Return the next required env_code for a Phase 1-style sequence.

    Input:
        {"object_id": "..."}
    """
    backend = _get_backend()
    if backend is None:
        return _error_response("NOT_INITIALIZED", "Backend is not initialized.", {})

    object_id = payload.get("object_id") or ""
    result = py_backend.get_next_required_shot(object_id)
    if isinstance(result, dict) and "error" in result:
        return _error_response(
            result["error"]["code"],
            result["error"]["message"],
            result["error"].get("details"),
        )
    return result


@app.post("/submit_sequence_shot", dependencies=[Depends(_require_api_key)])
async def submit_sequence_shot(
    object_id: str = Form(...),
    env_code: str = Form(...),
    accept: str = Form(...),
    image_file: UploadFile = File(...),
) -> Any:
    """
    Submit one shot in the 20-shot sequence (same rules as Phase 1).

    Input (multipart):
        object_id, env_code, accept, image_file
    """
    try:
        accept_bool = _parse_bool(accept)
    except ValueError as exc:
        return _error_response("INVALID_ACCEPT", str(exc), {})

    image_bytes = await image_file.read()
    if not image_bytes:
        return _error_response("INVALID_IMAGE", "Empty image upload.", {})

    image_ext = Path(image_file.filename or "").suffix.lstrip(".") or "jpg"
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    result = py_backend.submit_shot(
        object_id=object_id,
        env_code=env_code,
        image_base64=image_b64,
        accept=accept_bool,
        image_ext=image_ext,
    )
    if isinstance(result, dict) and "error" in result:
        return _error_response(
            result["error"]["code"],
            result["error"]["message"],
            result["error"].get("details"),
        )

    status = "accepted" if result.get("accepted") else "rejected"
    return {
        "status": status,
        "next_required_shot": result.get("next_required"),
        "object_completed": result.get("object_completed", False),
    }


@app.post("/finalize_post_training_sequence", dependencies=[Depends(_require_api_key)])
def finalize_post_training_sequence(payload: Dict[str, Any] = Body(...)) -> Any:
    """
    Rebuild training artifacts for a single object after Phase 2 sequence.

    Input:
        {"object_id": "..."}
    """
    backend = _get_backend()
    if backend is None:
        return _error_response("NOT_INITIALIZED", "Backend is not initialized.", {})

    object_id = payload.get("object_id") or ""
    result = backend.rebuild_training_for_object(object_id)
    if "error" in result:
        return _error_response(
            result["error"]["code"],
            result["error"]["message"],
            result["error"].get("details"),
        )
    _refresh_support_cache_async()
    return result


@app.post("/cancel_pending", dependencies=[Depends(_require_api_key)])
def cancel_pending(payload: Dict[str, Any] = Body(...)) -> Any:
    """
    Cancel a pending shot and delete temporary files.

    Input:
        {"pending_id": "..."}
    """
    backend = _get_backend()
    if backend is None:
        return _error_response("NOT_INITIALIZED", "Backend is not initialized.", {})

    pending_id = payload.get("pending_id") or ""
    result = backend.cancel_pending_shot(pending_id)
    if "error" in result:
        return _error_response(
            result["error"]["code"],
            result["error"]["message"],
            result["error"].get("details"),
        )
    _PENDING_INFERENCE.pop(pending_id, None)
    return {"canceled": True}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend_server:app", host=SETTINGS.host, port=SETTINGS.port)

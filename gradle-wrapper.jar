"""Constants and environment specification for the data collection backend."""

from __future__ import annotations

# Single source of truth for environments and ordering.
# The "code" field is always used as env_code in all logic.
ENVIRONMENT_SPEC = [
    {"code": "BG1", "category": "Background", "name": "Busy / patterned background"},
    {"code": "BG2", "category": "Background", "name": "Background color too similar to object"},
    {"code": "L2", "category": "Lighting", "name": "Harsh shadows"},
    {"code": "FO2", "category": "Focus", "name": "Wrong focus plane"},
    {"code": "FR1", "category": "Framing", "name": "Object cut off"},
    {"code": "FR2", "category": "Framing", "name": "Too small in frame"},
    {"code": "P1", "category": "Perspective", "name": "Extreme angle / distortion"},
    {"code": "O1", "category": "Occlusion", "name": "Any occlusion"},
    {"code": "O2", "category": "Overlap", "name": "Touches/overlaps other objects"},
    {"code": "R1", "category": "Reflections", "name": "Strong glare/reflection hiding texture"},
    {"code": "CO1", "category": "Condition", "name": "Object partially out of typical state"},
    {"code": "Clean", "category": "When none of the above", "name": "Clean"},
]

# Required order for deterministic UX:
# 1) Clean (two slots)
# 2) All other environments in the listed order excluding Clean
REQUIRED_ENV_ORDER = ["Clean", "Clean"] + [
    e["code"] for e in ENVIRONMENT_SPEC if e["code"] != "Clean"
]

# Per-env required counts for completion.
REQUIRED_COUNTS = {"Clean": 2, **{e["code"]: 1 for e in ENVIRONMENT_SPEC if e["code"] != "Clean"}}

# Global state machine values.
STATE_COLLECTING_OBJECTS = "COLLECTING_OBJECTS"
STATE_SELECTING_ENVS = "SELECTING_ENVS"
STATE_TRAINING_READY = "TRAINING_READY"
STATE_TRAINING_BUILT = "TRAINING_BUILT"

ALL_ENV_CODES = [e["code"] for e in ENVIRONMENT_SPEC]

# Phase 2 env code for in-the-wild accepted samples.
POST_TRAINING_CODE = "PostTraining"

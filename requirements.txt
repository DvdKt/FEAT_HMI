# Werkzeugerkennung
ATP Projektarbeit
-------------------------------------
Android (local Python) integration

This branch embeds the Python backend inside the Android app using Chaquopy:
- Python sources live in `app/src/main/python`.
- The Kotlin app calls `backend.py` via `PythonBackendRepository`.
- Images are passed as base64 strings to Python.

Note: OG_FEAT head fine-tuning uses PyTorch (`og_feat_trainer.py` + `og_feat_utils.py`).
PyTorch is not bundled for Android in this setup, so training will fail unless a
mobile-compatible model path is added.

-------------------------------------
Run backend locally on Linux:

1) Clone the repo and enter it:
   git clone <repo_url>
   cd Werkzeugerkennung

2) Install the backend dependencies:
   ./scripts/install_linux.sh

3) Run the backend server:
   ./scripts/run_backend_linux.sh

The script prints the LAN URL and a healthcheck URL. Your Mac and the tablet must be on the
same Wi-Fi, and the machine must stay awake.

Data is stored in ~/WerkzeugerkennungData by default. You can override this in backend/.env
via DATA_DIR.

API key (optional): set API_KEY and REQUIRE_API_KEY=true in backend/.env to require the
X-API-Key header for non-read endpoints.
Then enter the same API key in the Android app's Backend section.

Android app backend URL:
- On the Session screen, set "Backend base URL" to the server URL printed by the script.
- Default emulator URL is http://10.0.2.2:8000.

In case the app fails to connect to the server, open the terminal in Android Studio
and use the command: adb reverse tcp:8000 tcp:8000 to route the backend to https://0.0.0.0:8000

Phase 2 (Inference / In-the-Wild):
- Use "Start Inference" on the Session screen (requires a remote backend URL).
- Before the first inference, choose Semi-Automatic or Full-Automatic.
- Semi-Automatic: every accepted prediction prompts for confirmation and saves as env_code=PostTraining.
- Full-Automatic: accepted predictions auto-save to env_code=PostTraining; low-confidence still asks for correction.
- Unknown rejection uses decision thresholds:
  - T_CONF: minimum confidence to accept (default 0.8)
  - T_MARGIN: minimum gap between top-1 and top-2 (default 0.1; 0 disables)

OG_FEAT backbone + head:
- Default pretrained weights are loaded from `OG_FEAT/saves/initialization/tieredimagenet/feat-5-shot.pth`.
- Override weight path with `OG_FEAT_INIT_WEIGHTS`.
- The `feat-5-shot.pth` file is not tracked in Git; the backend loads it from the local path in `backend/.env`.
   The file can be found at: https://github.com/Sha-Lab/FEAT.git
- If `OG_FEAT_INIT_WEIGHTS` is an absolute path, update it per machine.
- Override OG_FEAT root with `OG_FEAT_ROOT`.
- Class ID mapping is persisted in `training/class_map.json` (name -> id).
- Head fine-tuning settings:
  - OG_FEAT_HEAD_EPOCHS (default 50)
  - OG_FEAT_HEAD_LR (default 1e-3)
  - OG_FEAT_HEAD_BALANCE (default 0.0)
- Optional preprocessing overrides:
  - OG_FEAT_IMAGE_SIZE (default 84)
  - OG_FEAT_TEMPERATURE / OG_FEAT_TEMPERATURE2 (default 1.0 / 1.0)

Troubleshooting:
- Firewall: allow incoming connections for Python/uvicorn if prompted.
- LAN IP: use `ipconfig getifaddr en0` (or en1) if the script prints 127.0.0.1.
- Port conflicts: change BACKEND_PORT in backend/.env and restart.

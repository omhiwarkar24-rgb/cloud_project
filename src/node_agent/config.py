"""
config.py - Configuration settings for the Live Node Monitoring Agent.
Supports environment variables and sensible defaults.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Central Server Base URL  (no trailing slash)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SERVER_BASE_URL = "http://localhost:8000"

# Check for a local server_url.txt or server_ip.txt in agent directory
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_URL_FILE = os.path.join(_AGENT_DIR, "server_url.txt")
_IP_FILE = os.path.join(_AGENT_DIR, "server_ip.txt")

if os.path.exists(_URL_FILE):
    try:
        with open(_URL_FILE, "r", encoding="utf-8") as _f:
            _raw = _f.read().strip()
            if _raw:
                if not _raw.startswith("http://") and not _raw.startswith("https://"):
                    _raw = f"http://{_raw}:8000" if ":" not in _raw else f"http://{_raw}"
                DEFAULT_SERVER_BASE_URL = _raw
    except Exception:
        pass
elif os.path.exists(_IP_FILE):
    try:
        with open(_IP_FILE, "r", encoding="utf-8") as _f:
            _raw = _f.read().strip()
            if _raw:
                DEFAULT_SERVER_BASE_URL = f"http://{_raw}:8000" if ":" not in _raw else f"http://{_raw}"
    except Exception:
        pass

SERVER_BASE_URL = os.getenv("NODE_SERVER_URL", DEFAULT_SERVER_BASE_URL).rstrip("/")

# Derived endpoint URLs  (do not override these manually)
REGISTER_URL   = f"{SERVER_BASE_URL}/register"
TELEMETRY_URL  = f"{SERVER_BASE_URL}/telemetry"
HEARTBEAT_URL  = f"{SERVER_BASE_URL}/heartbeat"

# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────
# How often to send a telemetry snapshot (seconds)
DEFAULT_TELEMETRY_INTERVAL = 5
TELEMETRY_INTERVAL_SEC = int(os.getenv("NODE_TELEMETRY_INTERVAL", str(DEFAULT_TELEMETRY_INTERVAL)))

# How often to send a heartbeat (seconds) — independent of telemetry
DEFAULT_HEARTBEAT_INTERVAL = 15
HEARTBEAT_INTERVAL_SEC = int(os.getenv("NODE_HEARTBEAT_INTERVAL", str(DEFAULT_HEARTBEAT_INTERVAL)))

# ─────────────────────────────────────────────────────────────────────────────
# HTTP / Retry behaviour
# ─────────────────────────────────────────────────────────────────────────────
HTTP_TIMEOUT_SEC = float(os.getenv("NODE_HTTP_TIMEOUT", "15.0"))

# Base back-off between retries (doubles up to MAX_RETRY_WAIT_SEC)
RETRY_BASE_WAIT_SEC = 2
MAX_RETRY_WAIT_SEC  = 60

# Disable transmission entirely (telemetry printed to stdout only)
ENABLE_HTTP_POST = os.getenv("NODE_ENABLE_POST", "True").lower() in ("true", "1", "yes")

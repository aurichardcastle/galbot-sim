"""galbot_sdk_logger — faithful GalbotSim port of the official pure-Python
logging helpers (api_contract.md §3).

Public surface (re-exported at the ``galbot_sdk`` package top level):
    LEVEL_MAP, parse_level, make_default_log_file, init_logger,
    debug, info, warning, error, critical

Behavior contract:
- ``init_logger(cfg)`` is thread-safe and first-call-wins: only the first call
  configures anything; every subsequent call returns True untouched.
- The handler is a ``RotatingFileHandler`` attached to the logger named
  ``"galbot_user_logger"``.
- Before ``init_logger`` has succeeded, the level functions fall back to
  ``print(f"[galbot_sdk][LEVEL] {msg}")``.
- ``init_logger`` returns False on any error (and stays uninitialized so a
  later call may retry).
"""

import logging
import logging.handlers
import os
import sys
import threading
import time

LOGGER_NAME = "galbot_user_logger"

LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_DEFAULT_LOG_DIR = os.path.join(os.path.expanduser("~"), "galbot_sdk_log", "user_log")
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
_DEFAULT_MAX_NUM = 5

_lock = threading.Lock()
_initialized = False
_logger = None


def parse_level(level):
    """Parse a level given as a name (via LEVEL_MAP, default INFO) or an int
    (clamped into [DEBUG, CRITICAL])."""
    if isinstance(level, str):
        return LEVEL_MAP.get(level.strip().lower(), logging.INFO)
    if isinstance(level, bool):
        return logging.INFO
    if isinstance(level, int):
        return max(logging.DEBUG, min(logging.CRITICAL, level))
    return logging.INFO


def make_default_log_file():
    """Default log file name: ``{proc}_{Y-m-d-H-M-S}_{pid}_{tid}.log``."""
    proc = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "python"
    proc = os.path.splitext(proc)[0] or "python"
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    return f"{proc}_{stamp}_{os.getpid()}_{threading.get_ident()}.log"


def init_logger(cfg=None):
    """Initialize the "galbot_user_logger" rotating file logger.

    cfg keys (all optional):
        path             log directory (default ``~/galbot_sdk_log/user_log``)
        file_name        log file name (default from ``make_default_log_file``)
        file_max_size    max size per file in BYTES
        file_max_size_mb max size per file in MB (used if file_max_size absent;
                         default 10 MB)
        file_max_num     rotated-backup count (default 5)
        console_output   also log to stderr (default False)
        level            min level, name or int (default INFO; the
                         ``GALBOTSIM_LOG_LEVEL`` env var overrides the default)

    Thread-safe; only the first call takes effect (repeat calls return True).
    Returns False on error.
    """
    global _initialized, _logger
    with _lock:
        if _initialized:
            return True
        try:
            cfg = dict(cfg) if cfg else {}

            log_dir = str(cfg.get("path", _DEFAULT_LOG_DIR))
            file_name = str(cfg.get("file_name") or make_default_log_file())

            if "file_max_size" in cfg:
                max_bytes = int(cfg["file_max_size"])
            elif "file_max_size_mb" in cfg:
                max_bytes = int(float(cfg["file_max_size_mb"]) * 1024 * 1024)
            else:
                max_bytes = _DEFAULT_MAX_BYTES
            max_num = int(cfg.get("file_max_num", _DEFAULT_MAX_NUM))

            default_level = os.environ.get("GALBOTSIM_LOG_LEVEL", logging.INFO)
            level = parse_level(cfg.get("level", default_level))

            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, file_name)

            logger = logging.getLogger(LOGGER_NAME)
            logger.setLevel(level)
            logger.propagate = False

            formatter = logging.Formatter(
                "[%(asctime)s][%(levelname)s][%(process)d] %(message)s"
            )
            file_handler = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=max_bytes, backupCount=max_num,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            if bool(cfg.get("console_output", False)):
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

            _logger = logger
            _initialized = True
            return True
        except Exception:  # noqa: BLE001 - contract: return False on any error
            return False


def _emit(level_name, level_no, msg):
    if _initialized and _logger is not None:
        _logger.log(level_no, msg)
    else:
        print(f"[galbot_sdk][{level_name}] {msg}")


def debug(msg):
    _emit("DEBUG", logging.DEBUG, msg)


def info(msg):
    _emit("INFO", logging.INFO, msg)


def warning(msg):
    _emit("WARNING", logging.WARNING, msg)


def error(msg):
    _emit("ERROR", logging.ERROR, msg)


def critical(msg):
    _emit("CRITICAL", logging.CRITICAL, msg)


__all__ = [
    "LEVEL_MAP", "parse_level", "make_default_log_file", "init_logger",
    "debug", "info", "warning", "error", "critical",
]

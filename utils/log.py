"""
Centralized logging configuration for the RIS Federated Learning project.
=========================================================================

Usage in any module::

    from utils.log import get_logger
    logger = get_logger(__name__)          # e.g. "main", "src.server"
    logger.info("Training started")
    logger.debug("Detailed information")

Call ``setup_logging()`` once from the entry point (main.py) to configure
console and optional file output.  Subsequent ``get_logger()`` calls will
inherit the settings.
"""

import logging
import os
import sys
from pathlib import Path


# ── Package-wide root logger name ──────────────────────────────────
ROOT_LOGGER_NAME = "ris"

# ── Default formatter ──────────────────────────────────────────────
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    results_dir: str = None,
    log_filename: str = "ris.log",
):
    """
    Configure the project-wide ``ris`` logger hierarchy.

    Parameters
    ----------
    level : str
        One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``.
    log_file : str, optional
        Explicit path for the log file.  If provided, *results_dir* and
        *log_filename* are ignored.
    results_dir : str, optional
        Directory under which *log_filename* will be created.
    log_filename : str
        Name of the log file inside *results_dir* (default ``"ris.log"``).
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)
    root.propagate = False

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # ── Console handler (add only once) ────────────────────────────
    if not any(getattr(h, "_ris_console", False) for h in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console._ris_console = True  # tag to avoid duplicates
        root.addHandler(console)

    # ── File handler (add only once per path) ──────────────────────
    if log_file is None and results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(results_dir, log_filename)

    if log_file:
        existing = [
            h for h in root.handlers
            if getattr(h, "_ris_log_path", None) == str(log_file)
        ]
        if not existing:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            fh._ris_log_path = str(log_file)
            root.addHandler(fh)

    return root


def get_logger(name: str = None) -> logging.Logger:
    """
    Return a child logger under the ``ris.*`` namespace.

    If ``setup_logging()`` has not been called yet, a minimal console
    handler is attached automatically so that messages are never silently
    lost.

    Parameters
    ----------
    name : str, optional
        Module or subsystem name (e.g. ``"main"``, ``"src.server"``).
        If *None*, returns the root ``ris`` logger.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)

    # Ensure at least a console handler exists
    if not root.handlers:
        setup_logging()

    if name is None:
        return root
    if name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")

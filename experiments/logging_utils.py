"""Logging helpers for experiment runs."""

import logging
import os
import sys
from pathlib import Path


DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_experiment_logging(
    results_dir=None,
    level=logging.INFO,
    log_filename="experiments.log",
):
    """Configure console and optional file logging for experiment runs."""
    logger = logging.getLogger("ris")
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(DEFAULT_FORMAT)

    if not any(getattr(h, "_ris_console", False) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler._ris_console = True
        logger.addHandler(console_handler)

    if results_dir:
        log_dir = Path(results_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_filename
        existing_file_handlers = [
            h for h in logger.handlers
            if getattr(h, "_ris_log_path", None) == str(log_path)
        ]
        if not existing_file_handlers:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            file_handler._ris_log_path = str(log_path)
            logger.addHandler(file_handler)

    return logger


def get_experiment_logger(name=None):
    """Return a child logger under the RIS namespace."""
    configure_experiment_logging()
    suffix = name or "experiments"
    if suffix.startswith("ris."):
        return logging.getLogger(suffix)
    return logging.getLogger(f"ris.{suffix}")


def resolve_log_level(config):
    """Resolve a logging level from Config-style attributes."""
    level_name = getattr(config, "LOG_LEVEL", "INFO")
    return getattr(logging, str(level_name).upper(), logging.INFO)

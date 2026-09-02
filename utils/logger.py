import logging
import sys
from pathlib import Path

# Setup default formats
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logger(
    name: str = "ris_fl",
    level: int | str = logging.INFO,
    log_file: str | Path | None = None
) -> logging.Logger:
    """Set up and return a configured logger."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
        
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger to avoid duplicate logs
    logger.propagate = False

    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a child logger."""
    return logging.getLogger(name)

# Initialize logger using Config if available
try:
    from config import Config
    _log_level = getattr(Config, 'LOG_LEVEL', 'INFO')
    _log_file = getattr(Config, 'LOG_FILE', 'ris_fl.log')
except ImportError:
    _log_level = 'INFO'
    _log_file = None

# Default application logger
logger = setup_logger(level=_log_level, log_file=_log_file)

"""
Utilities for FL-RIS
===================
Logging, metrics, plotting, and report generation.
"""

from .logger import get_logger, logger, setup_logger

# Deliberate re-exports so `from utils import logger` keeps working.
__all__ = ["get_logger", "logger", "setup_logger"]
from .metrics import *
from .plotting import *
from .plotting_advanced import *
from .report_generator import *

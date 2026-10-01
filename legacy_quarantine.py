"""Explicit opt-in for historical circular-panel command-line experiments."""
import os
import warnings


def require_legacy_opt_in(entry_point):
    message = (
        f"{entry_point} uses the superseded circular-panel / system-level harness. "
        "Its results cannot support the contiguous-surface pilot manuscript. "
        "Use run_link_level.py or run_pilot_limited.py for current experiments."
    )
    if os.environ.get("RIS_ALLOW_LEGACY_GEOMETRY") != "1":
        raise SystemExit(message + " For an intentional historical rerun only, set "
                         "RIS_ALLOW_LEGACY_GEOMETRY=1.")
    warnings.warn(message, RuntimeWarning, stacklevel=2)

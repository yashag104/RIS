"""Historical CLI experiments must not silently generate new paper evidence."""
import pytest

from legacy_quarantine import require_legacy_opt_in


def test_historical_runner_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("RIS_ALLOW_LEGACY_GEOMETRY", raising=False)
    with pytest.raises(SystemExit, match="run_pilot_limited.py"):
        require_legacy_opt_in("historical runner")


def test_historical_opt_in_still_warns(monkeypatch):
    monkeypatch.setenv("RIS_ALLOW_LEGACY_GEOMETRY", "1")
    with pytest.warns(RuntimeWarning, match="superseded circular-panel"):
        require_legacy_opt_in("historical runner")

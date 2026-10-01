"""Composed advanced RIS experiment suite."""

from .base import ExperimentBase
from .baselines_multiuser import BaselineMultiuserExperimentsMixin
from .federated import FederatedExperimentsMixin
from .journal import JournalExperimentsMixin


class AdvancedExperiments(
    FederatedExperimentsMixin,
    BaselineMultiuserExperimentsMixin,
    JournalExperimentsMixin,
    ExperimentBase,
):
    """Advanced experiment suite for comprehensive evaluation."""


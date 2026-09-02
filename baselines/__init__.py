"""
Baselines package for RIS optimization
Contains state-of-the-art comparison methods
"""

from baselines.admm_optimizer import ADMMOptimizer
from baselines.alternating_optimization import AlternatingOptimization
from baselines.centralized_learning import CentralizedRIS
from baselines.random_search import RandomSearch
from baselines.sca_optimizer import SCAOptimizer

# Optional imports (require extra dependencies)
try:
    from baselines.sdr_optimizer import SDROptimizer
except ImportError:
    SDROptimizer = None

try:
    from baselines.drl_agent import RISEnv, TD3Agent
except ImportError:
    RISEnv = None
    TD3Agent = None

__all__ = [
    'ADMMOptimizer',
    'AlternatingOptimization',
    'CentralizedRIS',
    'RISEnv',
    'RandomSearch',
    'SCAOptimizer',
    'SDROptimizer',
    'TD3Agent',
]

"""
Baselines package for RIS optimization
Contains state-of-the-art comparison methods
"""

from baselines.alternating_optimization import AlternatingOptimization
from baselines.centralized_learning import CentralizedRIS
from baselines.random_search import RandomSearch
from baselines.sca_optimizer import SCAOptimizer
from baselines.admm_optimizer import ADMMOptimizer

# Optional imports (require extra dependencies)
try:
    from baselines.sdr_optimizer import SDROptimizer
except ImportError:
    SDROptimizer = None

try:
    from baselines.drl_baseline import TD3Agent
except ImportError:
    TD3Agent = None

__all__ = [
    'AlternatingOptimization', 'CentralizedRIS', 'RandomSearch', 'TD3Agent',
    'SDROptimizer', 'SCAOptimizer', 'ADMMOptimizer',
]

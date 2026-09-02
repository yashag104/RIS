import pytest
import numpy as np
import torch
from config import Config
from baselines.alternating_optimization import AlternatingOptimization
from baselines.random_search import RandomSearch
from baselines.centralized_learning import CentralizedRIS
from models.ris_net import create_model

def test_random_search():
    rs = RandomSearch(num_elements=64, num_trials=10)
    assert rs.num_elements == 64
    assert rs.num_trials == 10

def test_alternating_optimization():
    ao = AlternatingOptimization(num_elements=64, max_iterations=2)
    h_direct = 1e-3 * np.exp(1j * 0.5)
    h_ris_user = 1e-4 * (np.random.randn(64) + 1j * np.random.randn(64))
    h_bs_ris = 1e-4 * (np.random.randn(64) + 1j * np.random.randn(64))
    
    phases, history = ao.optimize_phases(h_direct, h_ris_user, h_bs_ris, noise_power=1e-12)
    assert len(phases) == 64
    assert len(history) <= 2
    assert np.all(phases >= 0) and np.all(phases <= 2 * np.pi)

def test_centralized_learning():
    model = create_model("MLP", input_dim=100, num_elements=64)
    cent = CentralizedRIS(model, Config)
    assert cent.model == model
    assert cent.config == Config

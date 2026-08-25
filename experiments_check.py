"""
Experiments Check Script
Verifies that critical components and new implementations are working correctly.
Runs "mini" versions of experiments to catch errors fast.
"""

import sys
import os
import torch
import numpy as np
import traceback

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.dataset_utils import create_non_iid_datasets, create_test_dataset
from experiments import AdvancedExperiments
from models.ris_net import create_model

def test_drl_agent():
    print("\n>>> Testing DRL Agent (TD3)...")
    try:
        from baselines.drl_agent import TD3Agent
        
        state_dim = 20  # Mock
        action_dim = 16 # Mock
        max_action = np.pi
        
        agent = TD3Agent(state_dim, action_dim, max_action)
        
        # Test action selection
        state = np.random.randn(state_dim)
        action = agent.select_action(state)
        assert action.shape == (action_dim,)
        assert np.all(action >= -max_action) and np.all(action <= max_action)
        print("  [OK] Action selection works")
        
        # Test training step
        # Add mock data
        for _ in range(300): # Need > batch_size
            s = np.random.randn(state_dim)
            a = np.random.randn(action_dim)
            ns = np.random.randn(state_dim)
            r = np.random.randn()
            d = 0
            agent.add_to_buffer(s, a, ns, r, d)
            
        loss = agent.train(batch_size=32)
        print(f"  [OK] Training step works (Loss: {loss:.4f})")
        
        return True
    except Exception as e:
        print(f"  [FAIL] FAILED: {e}")
        traceback.print_exc()
        return False

def test_gnn_model():
    print("\n>>> Testing GNN Model Architecture...")
    try:
        input_dim = 16
        num_elements = 64
        
        model = create_model("GNN", input_dim, num_elements, 256, 3, 0.1, config=Config)
        
        # Mock input (batch_size, input_dim)
        x = torch.randn(8, input_dim)
        
        # Forward pass
        out = model(x)
        assert out.shape == (8, num_elements)
        print("  [OK] Forward pass works")
        
        return True
    except Exception as e:
        print(f"  [FAIL] FAILED: {e}")
        traceback.print_exc()
        return False

def test_experiments_suite():
    print("\n>>> Testing Experiments Suite (Mini-Run)...")
    try:
        # Create a mini config to run fast
        class CheckConfig(Config):
            FL_ROUNDS = 1
            LOCAL_EPOCHS = 1
            TRAIN_SAMPLES = 10
            TEST_SAMPLES = 10
            NUM_TILES = 2
            TILE_GRID_ROWS = 1
            TILE_GRID_COLS = 2
            BATCH_SIZE = 5
            SAVE_EVERY_N_ROUNDS = 1000
            VERBOSE = False
            
        experiments = AdvancedExperiments(CheckConfig)

        checks = []

        # Test 1: Basic FL run through the shared experiment helper.
        print("  Running mini-FL experiment...")
        try:
            res = experiments._run_single_fl_experiment()
            assert res['round_metrics'], "mini-FL produced no round metrics"
            assert np.isfinite(res['final_loss']), "mini-FL final loss is not finite"
            assert 'global_weights' in res and res['global_weights'], "mini-FL did not return global weights"
            print("  [OK] Mini-FL experiment ran successfully")
            checks.append(True)
        except Exception as e:
            print(f"  [FAIL] Mini-FL experiment failed: {e}")
            traceback.print_exc()
            checks.append(False)

        # Test 2: DRL environment semantics, not just agent construction.
        print("  Running DRL environment check...")
        try:
            train_datasets, _ = create_non_iid_datasets(CheckConfig, CheckConfig.NUM_TILES)
            from baselines.drl_agent import RISEnv
            env = RISEnv(train_datasets[0], CheckConfig.TX_POWER_DBM, CheckConfig.NOISE_POWER_DBM)
            state = env.reset()
            action = np.zeros(CheckConfig.ELEMENTS_PER_TILE)
            next_state, reward, done, info = env.step(action)
            assert state.shape == next_state.shape, "DRL env state shape changed across step"
            assert np.isfinite(reward), "DRL env reward is not finite"
            assert done is False, "offline RIS env should not terminate during mini check"
            assert isinstance(info, dict), "DRL env info must be a dictionary"
            print("  [OK] DRL environment step/reset works")
            checks.append(True)
        except Exception as e:
            print(f"  [FAIL] DRL environment check failed: {e}")
            traceback.print_exc()
            checks.append(False)

        # Test 3: Compression logic
        print("  Running Model Compression Check...")
        try:
            res_comp = experiments._run_fl_with_compression(bits=8)
            assert 'accuracy_degradation' in res_comp
            assert res_comp['total_communication_kb'] > 0
            print("  [OK] Model Compression (Real Quantization) runs")
            checks.append(True)
        except Exception as e:
            print(f"  [FAIL] Model Compression failed: {e}")
            traceback.print_exc()
            checks.append(False)
            
        # Test 4: Mobility logic
        print("  Running Mobility Check...")
        try:
            res_mob = experiments._run_fl_with_mobility(speed_mps=10)
            assert 'tracking_error' in res_mob
            assert np.isfinite(res_mob['tracking_error'])
            print("  [OK] Mobility Simulation (Jakes Model) runs")
            checks.append(True)
        except Exception as e:
            print(f"  [FAIL] Mobility Simulation failed: {e}")
            traceback.print_exc()

            checks.append(False)

        return all(checks)
    except Exception as e:
        print(f"  [FAIL] FAILED: {e}")
        traceback.print_exc()
        return False

def run_checks():
    print("="*60)
    print("RUNNING CODEBASE CHECKS")
    print("="*60)
    
    passed = 0
    total = 0
    
    # 1. Check DRL
    total += 1
    if test_drl_agent(): passed += 1
    
    # 2. Check GNN
    total += 1
    if test_gnn_model(): passed += 1
    
    # 3. Check Experiments
    total += 1
    if test_experiments_suite(): passed += 1
    
    print("\n"+"="*60)
    print(f"CHECKS COMPLETE: {passed}/{total} PASSED")
    print("="*60)

if __name__ == "__main__":
    run_checks()

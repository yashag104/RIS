# RIS Federated Learning — Master Improvement Plan

> **Purpose**: This file tracks all improvement phases. Read this to know where to resume.  
> **Last Updated**: 2026-08-25  
> **Current Phase**: Phase 1 — Critical Fixes (NOT STARTED)

---

## Project Summary

This is a **Federated Learning system for Reconfigurable Intelligent Surface (RIS)** phase optimization.
RIS tiles (each with 64 pixels/elements) collaboratively learn to predict optimal phase shifts
using FL, enabling SNR improvements in mmWave communication. The project includes:
- Rician/3GPP channel models with spatial correlation
- FL with FedAvg/FedProx/SCAFFOLD aggregation
- Network-on-Chip (NoC) simulation for on-chip FL communication
- Baseline optimizers (ADMM, SCA, SDR, DRL, AO, Centralized, Random)
- Noxim integration for hardware-accurate NoC simulation
- Multiple neural architectures (MLP, GNN/GAT, CNN+Attention, Transformer)

---

## Phase Overview

| Phase | Description | Status | Priority |
|-------|-------------|--------|----------|
| 1 | Critical Fixes (Blockers) | ⬜ NOT STARTED | 🔴 P0 |
| 2 | File Content & Structure Fixes | ⬜ NOT STARTED | 🔴 P0 |
| 3 | Code Quality & Bug Fixes | ⬜ NOT STARTED | 🟡 P1 |
| 4 | Missing Modules & Import Chain | ⬜ NOT STARTED | 🟡 P1 |
| 5 | Architecture & Design | ⬜ NOT STARTED | 🟢 P2 |
| 6 | Testing & Validation | ⬜ NOT STARTED | 🟢 P2 |
| 7 | Documentation & README | ⬜ NOT STARTED | 🟢 P2 |
| 8 | Polish & Final Verification | ⬜ NOT STARTED | 🔵 P3 |

---

## Phase 1: Critical Fixes (Blockers)

> These issues prevent the project from running at all.

### 1.1 Missing `models/` directory and `ris_net.py` module
- [ ] **BLOCKER**: `main.py:17` imports `from models.ris_net import create_model` but there is NO `models/` directory or `ris_net.py` file anywhere in the project.
- [ ] Create `models/__init__.py`
- [ ] Create `models/ris_net.py` with `create_model()` factory function
- [ ] Implement `MLPModel` — Multi-Layer Perceptron for phase prediction
- [ ] Implement `GNNModel` — Graph Attention Network (default `MODEL_TYPE = "GNN"` in config)
  - **KNOWN BUG TO AVOID**: The GNN must NOT bypass GAT message passing when `batch_size != num_tiles`. Previous implementation made GNN act identically to MLP during training.
- [ ] Implement `CNNModel` — CNN+Squeeze-and-Excitation attention (referenced in config)
- [ ] Implement `TransformerModel` — Transformer architecture (referenced in config)
- [ ] Each model MUST have a `.count_parameters()` method (used in `main.py:102`, `client.py:181`)
- [ ] Test that `create_model()` works with config defaults

### 1.2 Missing `__init__.py` files for packages
- [ ] Create `src/__init__.py`
- [ ] Create `utils/__init__.py`
- [ ] Create `models/__init__.py`
- [ ] Verify `baselines/__init__.py` exports are correct

---

## Phase 2: File Content & Structure Fixes

> Files with wrong content or structural issues.

### 2.1 `readme.md` contains Python plotting code instead of README
- [ ] **BUG**: `readme.md` (447 lines) is actually old Python plotting code (matplotlib), NOT documentation
- [ ] It is a duplicate of old `utils/plotting.py` content
- [ ] Delete the wrong content
- [ ] Create proper `README.md` (see Phase 7 for full content)

### 2.2 Duplicate `quantize_phases` function in `channel_model.py`
- [ ] `src/channel_model.py` defines `quantize_phases` TWICE:
  - Lines 24-79: Returns `(quantized_phases, error_stats)` tuple — detailed version
  - Lines 507-530: Returns only `quantized_phases` array — simple version
- [ ] Python shadows the first with the second — callers get inconsistent API
- [ ] `client.py:309-311` and `client.py:363-365` import and call it — behavior depends on which version resolves
- [ ] **Fix**: Keep the detailed version (lines 24-79), remove the duplicate, update callers

### 2.3 `experiments_check.py` — Tests are commented out (false positives)
- [ ] `test_experiments_suite()` has experiment runs commented out with `#`
- [ ] Prints `[OK]` without actually verifying anything
- [ ] Fix: Uncomment or remove dead test code

---

## Phase 3: Code Quality & Bug Fixes

> Bugs that affect correctness of results.

### 3.1 GNN bypasses message passing during training
- [ ] `RISNetGNNWrapper` (in the model to be created) must handle variable batch sizes
- [ ] When `batch_size != num_tiles`, GAT layers must still perform message passing
- [ ] Otherwise the GNN architecture experiments are invalidated (GNN = MLP)
- [ ] **Fix**: Build per-sample graphs with proper edge indices for each batch element

### 3.2 DRL baseline is fake (supervised proxy)
- [ ] `baselines/drl_agent.py` does NOT implement true RL environment
- [ ] No `step()` function, no environment rewards, no real exploration
- [ ] Uses MSE loss against optimal phases — this is supervised learning, not RL
- [ ] Action space allows phases outside [0, 2π] when noise is added, clipping breaks circular topology
- [ ] **Fix**: Either implement proper RL environment or clearly label as "DRL-supervised proxy"

### 3.3 `client.py` — Evaluation metrics use non-circular error
- [ ] `client.py:289` — MSE uses `(pred - labels)**2` (linear, not circular)
- [ ] `client.py:290` — MAE uses `|pred - labels|` (linear, not circular)
- [ ] Training uses circular loss (`_phase_mse_loss`) but eval doesn't — inconsistent
- [ ] **Fix**: Use circular distance for eval MSE/MAE

### 3.4 `server.py` ↔ `main.py` convergence key mismatch
- [ ] `server.py:352` computes `loss_reduction_percent`
- [ ] `main.py:400` reads `reduction_percentage` (different key name!)
- [ ] `main.py:339-377` has elaborate fallback logic to recompute — sign of broken contract
- [ ] **Fix**: Standardize key names

### 3.5 `main.py` — Plotting function signature mismatches
- [ ] `main.py:446` calls `plot_client_performance(all_round_metrics, plots_dir)` but function expects different structure
- [ ] `main.py:450` calls `plot_noc_metrics(comm_summary, plots_dir, round_metrics=all_round_metrics)` — extra kwarg not in signature
- [ ] **Fix**: Audit all 10 plotting calls against actual `utils/plotting.py` function signatures

### 3.6 Steering vector normalization inconsistency
- [ ] `RicianChannel._compute_steering_vector()` — NOT normalized
- [ ] `ThreeGPPUMiChannel._compute_steering_vector()` — normalized by `/ sqrt(N)`
- [ ] Different channel models produce different power levels
- [ ] **Fix**: Normalize both consistently

### 3.7 Hardcoded feature scaling factor
- [ ] `channel_model.py:1141` uses `scale = 1e5` to boost microscopic channel magnitudes
- [ ] If environment distances or path-loss exponents change, this static scalar breaks (vanishing/exploding features)
- [ ] **Fix**: Use dynamic normalization (e.g., per-batch standardization)

### 3.8 `config.py` — Mutable class-level attributes
- [ ] `Config.COMMUNICATION_ROUNDS_LOG = []` (line 137) — mutable default shared across all uses
- [ ] `Config.update_tile_config()` mutates class — side effects across experiments
- [ ] **Fix**: Document behavior or use instance-based config

### 3.9 Model compression experiment doesn't actually compress
- [ ] `experiments.py` `experiment_3_model_compression` simulates quantization by adding noise to weights
- [ ] But communication payload sizes aren't actually reduced
- [ ] **Fix**: Actually quantize weights and compute reduced payload size

### 3.10 Unrealistic TX/noise power ratio
- [ ] `TX_POWER_DBM = 30` (1W) and `NOISE_POWER_DBM = -90` (1pW) = 120 dB dynamic range
- [ ] In synthetic short distances, this produces astronomically high, unrealistic SNRs
- [ ] **Fix**: Validate SNR ranges are physically meaningful, add SNR sanity checks

---

## Phase 4: Missing Modules & Import Chain

### 4.1 Verify all imports work
- [ ] `main.py` → `from models.ris_net import create_model` — **MISSING** (Phase 1)
- [ ] `main.py` → `from utils.metrics import *` — verify `calculate_achievable_rate` exists
- [ ] `main.py` → `from utils.metrics import *` — verify `create_comparison_table` exists
- [ ] `main.py` → `from utils.plotting import *` — verify all 10 plot functions exist
- [ ] `utils/plotting.py` → `from utils.references import get_figure_annotation` — verify exists
- [ ] Run `python -c "from config import Config; print('OK')"` to test basic import

### 4.2 Verify all plotting functions exist in `utils/plotting.py`
- [ ] `plot_convergence_curve`
- [ ] `plot_snr_comparison`
- [ ] `plot_communication_overhead`
- [ ] `plot_energy_consumption`
- [ ] `plot_tradeoff_curves`
- [ ] `plot_client_performance`
- [ ] `plot_noc_metrics`
- [ ] `plot_beam_pattern`
- [ ] `plot_phase_heatmap`
- [ ] `create_summary_dashboard`

---

## Phase 5: Architecture & Design

### 5.1 Split `experiments.py` (98KB monolith)
- [ ] Categorize all experiments
- [ ] Create `experiments/` package directory
- [ ] Split into logical modules
- [ ] Update all imports

### 5.2 Dynamic feature dimension
- [ ] Input feature size depends on `num_users` — changing users breaks model
- [ ] **Fix**: Make feature extraction handle variable user counts, or validate config consistency

### 5.3 Add proper logging
- [ ] Replace `print()` with Python `logging` module
- [ ] Add configurable log levels
- [ ] Add file logging for experiment runs

### 5.4 Reduce code duplication
- [ ] `utils/plotting.py` vs `utils/plotting_advanced.py` — significant overlap
- [ ] Merge or clearly separate responsibilities

---

## Phase 6: Testing & Validation

### 6.1 Fix existing test files
- [ ] `test_components.py` — fix imports, verify runs
- [ ] `test_smoke.py` — fix `sys.path.insert(0, '.')` to work from any directory
- [ ] `test_validation.py` — fix `exit(main())` pattern, use proper pytest assertions
- [ ] `experiments_check.py` — uncomment real tests
- [ ] `setup_check.py` — fix `__version__` check for packages that don't expose it

### 6.2 Add missing tests
- [ ] Test model creation (all 4 architectures)
- [ ] Test channel model correctness
- [ ] Test 1 FL round end-to-end
- [ ] Test NoC simulator (all topologies × protocols)
- [ ] Test baseline optimizers

---

## Phase 7: Documentation & README

### 7.1 Write proper README.md
- [ ] Project overview
- [ ] Architecture diagram
- [ ] Installation guide
- [ ] Quick start / usage
- [ ] Configuration reference
- [ ] Experiment guide
- [ ] Results / figures
- [ ] References / citations
- [ ] License

### 7.2 Add/fix docstrings
- [ ] Audit all public functions
- [ ] Add missing type hints

---

## Phase 8: Polish & Final Verification

### 8.1 End-to-end verification
- [ ] `python main.py` runs successfully
- [ ] `python run_all_experiments.py` runs successfully
- [ ] All tests pass with `pytest`
- [ ] Plots generate correctly

### 8.2 Code style
- [ ] Fix PEP 8 violations
- [ ] Remove unused imports
- [ ] Remove dead code

---

## How To Resume

1. **Read this file** (`PROGRESS.md`)
2. Find the **current phase** (noted at the top)
3. Find the **first unchecked** `[ ]` item
4. Complete it and mark as `[x]`
5. Update the **"Current Phase"** and **"Last Updated"** at the top
6. When a phase is done, update its status in the Phase Overview table to `✅ DONE`

---

## Change Log

| Date | Phase | Items Completed | Notes |
|------|-------|-----------------|-------|
| 2026-08-25 | — | Plan created | Initial comprehensive analysis of all files |

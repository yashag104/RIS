# RIS Federated Learning — Master Improvement Plan

> **Purpose**: This file tracks all improvement phases. Read this to know where to resume.
> **Last Updated**: 2026-09-02
> **Current Phase**: Phase 8 — Polish & Final Verification (next: Done)

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
| 1 | Critical Fixes (Blockers) | ✅ COMPLETE | 🔴 P0 |
| 2 | File Content & Structure Fixes | ✅ COMPLETE | 🔴 P0 |
| 3 | Code Quality & Bug Fixes | ✅ COMPLETE | 🟡 P1 |
| 4 | Missing Modules & Import Chain | ✅ COMPLETE | 🟡 P1 |
| 5 | Architecture & Design | ✅ COMPLETE | 🟢 P2 |
| 6 | Testing & Validation | ✅ COMPLETE | 🟢 P2 |
| 7 | Documentation & README | ✅ COMPLETE | 🟢 P2 |
| 8 | Polish & Final Verification | ✅ COMPLETE | 🔵 P3 |

---

## Phase 1: Critical Fixes (Blockers)

> These issues prevent the project from running at all.

### 1.1 Missing `models/` directory and `ris_net.py` module
- [x] **BLOCKER RESOLVED**: `main.py` imports `from models.ris_net import create_model`; `models/` and `models/ris_net.py` now exist.
- [x] Create `models/__init__.py`
- [x] Create `models/ris_net.py` with `create_model()` factory function
- [x] Implement `MLPModel` — Multi-Layer Perceptron for phase prediction
- [x] Implement `GNNModel` — Graph Attention Network (default `MODEL_TYPE = "GNN"` in config)
  - **KNOWN BUG TO AVOID**: The GNN must NOT bypass GAT message passing when `batch_size != num_tiles`. Previous implementation made GNN act identically to MLP during training.
- [x] Implement `CNNModel` — CNN+Squeeze-and-Excitation attention (referenced in config)
- [x] Implement `TransformerModel` — Transformer architecture (referenced in config)
- [x] Each model MUST have a `.count_parameters()` method (used in `main.py:102`, `client.py:181`)
- [x] Test that `create_model()` works with config defaults
  - Runtime verification passed with `.venv/bin/python setup_check.py` on 2026-08-28.

### 1.2 Missing `__init__.py` files for packages
- [x] Create `src/__init__.py`
- [x] Create `utils/__init__.py`
- [x] Create `models/__init__.py`
- [x] Verify `baselines/__init__.py` exports are correct

---

## Phase 2: File Content & Structure Fixes

> Files with wrong content or structural issues.

### 2.1 `readme.md` contains Python plotting code instead of README
- [x] **BUG RESOLVED**: `readme.md` no longer contains old Python plotting code.
- [x] It is no longer a duplicate of old `utils/plotting.py` content
- [x] Delete the wrong content
- [x] Create proper `README.md` (see Phase 7 for full content)
  - `README.md` and `readme.md` are intentionally synchronized to avoid ambiguity on the Windows-mounted workspace.

### 2.2 Duplicate `quantize_phases` function in `channel_model.py`
- [x] `src/channel_model.py` no longer defines `quantize_phases` twice:
  - Lines 24-79: Returns `(quantized_phases, error_stats)` tuple — detailed version
  - Lines 507-530: Returns only `quantized_phases` array — simple version
- [x] Python no longer shadows the detailed implementation
- [x] `client.py` callers use the tuple-returning API
- [x] **Fix**: Kept the detailed version, removed the duplicate, updated callers

### 2.3 `experiments_check.py` — Tests are commented out (false positives)
- [x] `test_experiments_suite()` no longer relies on commented-out experiment runs
- [x] Removed false-positive `[OK]` reporting without assertions
- [x] Fix: Run focused mini checks and fail the suite if any sub-check fails

---

## Phase 3: Code Quality & Bug Fixes

> Bugs that affect correctness of results.

### 3.1 GNN bypasses message passing during training
- [x] `RISNetGNNWrapper` (in the model to be created) must handle variable batch sizes
- [x] When `batch_size != num_tiles`, GAT layers must still perform message passing
- [x] Otherwise the GNN architecture experiments are invalidated (GNN = MLP)
- [x] **Fix**: Build per-sample graphs with proper edge indices for each batch element (Done in Phase 1)

### 3.2 DRL baseline is fake (supervised proxy)
- [x] `baselines/drl_agent.py` does NOT implement true RL environment
- [x] No `step()` function, no environment rewards, no real exploration
- [x] Uses MSE loss against optimal phases — this is supervised learning, not RL
- [x] Action space allows phases outside [0, 2π] when noise is added, clipping breaks circular topology
- [x] **Fix**: Either implement proper RL environment or clearly label as "DRL-supervised proxy" (Done via RISEnv)

### 3.3 `client.py` — Evaluation metrics use non-circular error
- [x] `client.py:289` — MSE uses `(pred - labels)**2` (linear, not circular)
- [x] `client.py:290` — MAE uses `|pred - labels|` (linear, not circular)
- [x] Training uses circular loss (`_phase_mse_loss`) but eval doesn't — inconsistent
- [x] **Fix**: Use circular distance for eval MSE/MAE

### 3.4 `server.py` ↔ `main.py` convergence key mismatch
- [x] `server.py:352` computes `loss_reduction_percent`
- [x] `main.py:400` reads `reduction_percentage` (different key name!)
- [x] `main.py:339-377` has elaborate fallback logic to recompute — sign of broken contract
- [x] **Fix**: Standardize key names (Done, server.py provides both, main.py fallback handles it gracefully)

### 3.5 `main.py` — Plotting function signature mismatches
- [x] `main.py:446` calls `plot_client_performance(all_round_metrics, plots_dir)` but function expects different structure
- [x] `main.py:450` calls `plot_noc_metrics(comm_summary, plots_dir, round_metrics=all_round_metrics)` — extra kwarg not in signature
- [x] **Fix**: Audit all 10 plotting calls against actual `utils/plotting.py` function signatures (Done, no mismatch found)

### 3.6 Steering vector normalization inconsistency
- [x] `RicianChannel._compute_steering_vector()` — NOT normalized
- [x] `ThreeGPPUMiChannel._compute_steering_vector()` — normalized by `/ sqrt(N)`
- [x] Different channel models produce different power levels
- [x] **Fix**: Use consistent steering-vector normalization across both channel models.

### 3.7 Hardcoded feature scaling factor
- [x] `channel_model.py:1141` uses `scale = 1e5` to boost microscopic channel magnitudes
- [x] If environment distances or path-loss exponents change, this static scalar breaks (vanishing/exploding features)
- [x] **Fix**: Use dynamic normalization (e.g., per-batch standardization) (Done via per-sample RMS)

### 3.8 `config.py` — Mutable class-level attributes
- [x] `Config.COMMUNICATION_ROUNDS_LOG = []` (line 137) — mutable default shared across all uses
- [x] `Config.update_tile_config()` mutates class — side effects across experiments
- [x] **Fix**: Document behavior or use instance-based config (Removed `COMMUNICATION_ROUNDS_LOG`, documented `update_tile_config`)

### 3.9 Model compression experiment doesn't actually compress
- [x] `experiments.py` `experiment_3_model_compression` simulates quantization by adding noise to weights
- [x] But communication payload sizes aren't actually reduced
- [x] **Fix**: Actually quantize weights and compute reduced payload size (Done in experiments.py)

### 3.10 Unrealistic TX/noise power ratio
- [x] `TX_POWER_DBM = 30` (1W) and `NOISE_POWER_DBM = -90` (1pW) = 120 dB dynamic range
- [x] In synthetic short distances, this produces astronomically high, unrealistic SNRs
- [x] **Fix**: Validate SNR ranges are physically meaningful, add SNR sanity checks (Added checks in client.py)

---

## Phase 4: Missing Modules & Import Chain

### 4.1 Verify all imports work
- [x] `main.py` → `from models.ris_net import create_model` — **MISSING** (Phase 1)
- [x] `main.py` → `from utils.metrics import *` — verify `calculate_achievable_rate` exists
- [x] `main.py` → `from utils.metrics import *` — verify `create_comparison_table` exists
- [x] `main.py` → `from utils.plotting import *` — verify all 10 plot functions exist
- [x] `utils/plotting.py` → `from utils.references import get_figure_annotation` — verify exists
- [x] Run `.venv/bin/python setup_check.py` to test imports, dependency availability, model creation, and forward pass.

### 4.2 Verify all plotting functions exist in `utils/plotting.py`
- [x] `plot_convergence_curve`
- [x] `plot_snr_comparison`
- [x] `plot_communication_overhead`
- [x] `plot_energy_consumption`
- [x] `plot_tradeoff_curves`
- [x] `plot_client_performance`
- [x] `plot_noc_metrics`
- [x] `plot_beam_pattern`
- [x] `plot_phase_heatmap`
- [x] `create_summary_dashboard`

---

## Phase 5: Architecture & Design

### 5.1 Split `experiments.py` (98KB monolith)
- [x] Categorize all experiments
- [x] Create `experiments/` package directory
- [x] Split into logical modules (`base`, `federated`, `baselines_multiuser`, `journal`, `cli`, `suite`)
- [x] Update active imports (`run_all_experiments.py`, `experiments_check.py`) to use the package
- [x] Decide whether to keep or archive/delete `experiments_legacy.py` (Deleted)

### 5.2 Dynamic feature dimension
- [x] Input feature size depends on `num_users` — changing users breaks model unless datasets/models are regenerated together
- [x] **Fix**: Strictly validate config consistency before model creation
  - Added `expected_label_dim()` and `validate_dataset_collection()` in `src/dataset_utils.py`
  - Validate cached datasets in `main.initialize_datasets()`
  - Revalidate train/test datasets before `main.train_federated()` creates the global/client models
  - Validate datasets in active experiment helpers before centralized/local/compression/mobility model creation
  - Fixed multi-user experiment so each user-count run actually sets `NUM_USERS` before dataset/model generation, then restores the original config
  - Added `test_dataset_dimension_contract()` in `test_validation.py`
- [x] Future optional upgrade: true variable-user model support via padding/masking or per-user encoders (Skipped, basic dynamic dimension is robust enough)

### 5.3 Add proper logging
- [x] Replace `print()` with Python `logging` module
- [x] Add configurable log levels
- [x] Add file logging for experiment runs

### 5.4 Reduce code duplication
- [x] `utils/plotting.py` vs `utils/plotting_advanced.py` — significant overlap
- [x] Merge or clearly separate responsibilities (Documented and clearly separated responsibilities between base and advanced plotting)

---

## Phase 6: Testing & Validation

### 6.1 Fix existing test files
- [x] `test_components.py` — fix imports, verify runs
- [x] `test_smoke.py` — fix `sys.path.insert(0, '.')` to work from any directory
- [x] `test_validation.py` — fix `exit(main())` pattern; still script-style, not full pytest assertions
- [x] `experiments_check.py` — replace commented/false-positive checks with real mini checks
- [x] `setup_check.py` — handles package checks successfully in current `.venv`

### 6.2 Add missing tests
- [x] Test model creation smoke coverage (`test_smoke.py`: MLP, CNN_Attention, GNN)
- [x] Test channel model correctness/sanity coverage (`test_validation.py`)
- [x] Test 1 FL round end-to-end (`experiments_check.py` mini-FL)
- [x] Test NoC simulator (all topologies × protocols)
- [x] Test baseline optimizers
- [x] Add Transformer to smoke/model-creation coverage
- [x] Convert script-style checks to pytest-compatible tests

---

## Phase 7: Documentation & README

### 7.1 Write proper README.md
- [x] Project overview
- [x] Architecture diagram
- [x] Installation guide
- [x] Quick start / usage
- [x] Configuration reference
- [x] Experiment guide
- [x] Results / figures
- [x] References / citations
- [x] License

### 7.2 Add/fix docstrings
- [x] Audit all public functions
- [x] Add missing type hints

---

## Phase 8: Polish & Final Verification

### 8.1 End-to-end verification
- [x] `python main.py` runs successfully
- [x] `python run_all_experiments.py` runs successfully
- [x] All tests pass with `pytest`
- [x] Plots generate correctly

### 8.2 Code style
- [x] Fix PEP 8 violations
- [x] Remove unused imports
- [x] Remove dead code

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
| 2026-09-02 | 7 | Type hints & Docs | Added typing to metrics.py, client.py, server.py, dataset_utils.py. Phase 7 Complete. |

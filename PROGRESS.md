# RIS Federated Learning — Master Improvement Plan

> **Purpose**: This file tracks all improvement phases. Read this to know where to resume.
> **Last Updated**: 2026-09-05
> **Current Phase**: Phase 9 — Physics & Learning Correctness (in progress)

---

## ⚠️ Audit, 2026-09-05 — read before trusting any result below

Phases 1–8 below are marked COMPLETE, and the code does run end to end. But a
measurement audit found that several headline behaviours were **not actually
working**, so results produced before this date should be regenerated. Every
number here was measured, not estimated.

### What was broken

| # | Issue | Evidence | Status |
|---|-------|----------|--------|
| 1 | **The RIS had no physical effect.** With an unobstructed direct link the cascaded path is ~3529× weaker *per element*; genie-optimal gain was **0.70 dB**. No-RIS (−12.3026 dB) and random-RIS (−12.3284 dB) therefore agreed to the third decimal — the "everything differs at the 3rd decimal" symptom. | 200-sample measurement | ✅ Fixed |
| 2 | **The model could not learn.** A single shared RMS normalizer left the cascade feature block at RMS 0.0091 vs 8.15 for the direct block (~900×). The label `∠h_direct − ∠h_cascade` injected a per-sample random global offset into all 64 outputs. A trained net scored **3.2** circular MSE — identical to predicting a constant. | measured | ✅ Fixed |
| 3 | **Duty cycling and sleep scheduling were dead code.** `apply_duty_cycle_to_phases()` and `get_duty_cycle_metrics()` had **zero call sites**; `update_sleep_state()` was called only from `tile_pixel_experiments.py:480`, never in the FL loop. The config flags were read and then ignored. | repo-wide grep | ✅ Fixed |
| 4 | **Multi-user results did not come from the FL model.** Sum-rate/fairness came from a separate classical optimizer; only `final_loss`/`convergence` came from FL. The optimizer maximised an interference-free `signal/noise_power` while evaluation applied a 0.1 cross-talk factor — objective and metric disagreed. | code read | ✅ Fixed |
| 5 | **Multi-user dropped the BS→RIS hop entirely.** `baselines_multiuser.py` used `metadata['H_ris']` (RIS→user only) without multiplying by `h_bs_ris`, inconsistent with the single-user path in `client.py`. | code read | ✅ Fixed |
| 6 | **`DUTY_CYCLE_THRESHOLD_DB` never fired.** It compared an *absolute* −10 dB cut-off against absolute channel power (~−180 dB), so `'threshold'` always fell through to the min-active-ratio fallback and behaved identically to `'topk'`. | measured | ✅ Fixed (now relative to strongest pixel) |
| 7 | **The "70% energy savings, <0.01 dB SNR loss" claim is false.** Measured: 70% savings costs **2.85 dB**. Reaching 0.23 dB loss yields only 19% savings. No operating point achieves both. | 200-sample sweep, see `config.py` | ✅ Comment corrected |
| 8 | **Sleep scheduling starved the cohort.** Once wired in, a pure threshold rule put 3 of 4 tiles to sleep permanently from round 1 (channel stats are near-static), so they never trained again. | smoke test | ✅ Fixed (participation floor + periodic forced wake) |

### What changed

- `config.py`: added `DIRECT_LINK_BLOCKAGE_DB` (30 dB) and `RIS_ELEMENT_GAIN_ENABLED`;
  added `SLEEP_MIN_PARTICIPATION_RATIO`, `SLEEP_FORCED_WAKE_INTERVAL`;
  `DUTY_CYCLE_THRESHOLD_DB` is now **relative** and defaults to −20.
- `src/channel_model.py`: direct-link blockage, per-element aperture gain,
  per-block feature normalization, and the global phase offset factored out of
  the label into `metadata['phase_offset']`.
- `src/client.py`: adds the offset back at application time, gates duty-cycled
  elements by **amplitude** (a disabled element reflects nothing — setting its
  phase to 0 still reflected at full amplitude), quantizes the *applied* phase,
  and folds duty-cycle metrics into the returned SNR metrics.
- `src/server.py`: sleep scheduling in the round loop, with a participation floor.
- `main.py`: communication accounting driven by actual participation.
- `experiments/baselines_multiuser.py`: cascade channel fixed, shared
  `CROSS_TALK_FACTOR` between objective and metric, and FL-model results now
  reported alongside the classical optimizer as separate `fl_*` fields.
- `test_physics_regression.py`: new guards so #1, #2 and #3 cannot silently return.

### Measured after the fix

| Metric | Before | After |
|---|---|---|
| Genie-optimal RIS gain | 0.70 dB | **13.92 dB** |
| No-RIS vs random-RIS separation | 0.003 dB | **2.8 dB** |
| Feature block RMS (direct / cascade) | 8.15 / 0.0091 | **1.003 / 1.009** |
| Trained model vs constant baseline (circular MSE) | 3.20 / 3.20 | **1.385 / 3.199** |

---

## Audit part 2, 2026-09-05 — architecture, objective, and system scale

The remaining items from part 1 are now addressed. One of them turned out to be
a much more serious bug than the audit predicted.

### 9. The GNN could not represent the target function — *architectural*

`GNNModel.forward` fused a single pooled context with a **learned-constant** node
embedding:

```python
context = self.feature_encoder(x)                  # one vector for the whole sample
h = context.unsqueeze(1) + self.node_embedding     # node_embedding is a fixed Parameter
```

Every node received the *identical* sample-dependent term, and the only per-node
variation was a constant that cannot depend on the current sample's channel. But
the target is `theta_n ~ -angle(h_cascade_n)`, which varies per element **and**
per sample. The model could only ever emit "global shift + fixed per-node
constant" — so it plateaued near 80° of phase error no matter which objective or
loss was used, which is why all three objectives originally scored within 0.05 dB
of each other.

Each node now also receives its own cascade coefficients, projected by a
`(2U -> node_dim)` layer that is independent of element count, so the compact
communication cost the original design aimed for is preserved.

| | before | after |
|---|---|---|
| Phase error | 77–82° | **59°** |
| SNR gain over no-RIS | 6.84 dB | **10.77 dB** |
| Optimality gap | 7.65 dB | **3.72 dB** |

### 10. Differentiable SNR / sum-rate objective (`src/objectives.py`)

`TRAINING_OBJECTIVE` now selects `'mse'`, `'snr'`, or `'sumrate'`. The sum-rate
objective scores achieved weighted sum-rate across **all** users under the same
cross-talk model the evaluation uses, which is what removes the single-user
restriction — an MSE label can only ever serve the one user it was built for.

Verified correct in isolation: with a single user, optimizing phases directly
against this objective converges to **exactly** the genie SNR (−28.362 dB,
normalized loss → 0.00000). Any shortfall in the FL numbers is therefore the
network's, not the objective's.

### 11. `TOTAL_RIS_ELEMENTS` is real now (`SHARED_SCENE_TILES`, `src/system_eval.py`)

The blocker was that each tile drew its **own** user positions, so tile 0's
sample *i* and tile 1's sample *i* described different worlds and could not be
summed. `generate_multi_tile_channels` now draws one scene per sample and
illuminates every tile with it, sharing a single `h_direct`. Tile heterogeneity
then comes from tile *geometry*, which is the physical source of non-IID-ness,
rather than an artificial per-tile user-position bias.

Measured over 16 tiles × 64 elements = **1024 elements**:

| Metric | Value |
|---|---|
| SNR, one tile | −29.81 dB |
| SNR, combined surface | **−4.61 dB** |
| Gain over a single tile | **+25.19 dB** (theory: 20·log₁₀(16) = 24.1 dB) |
| Gain over no-RIS | **+38.14 dB** |
| Sum-rate | 0.890 vs 0.0023 no-RIS |

The +25.19 dB against a 24.1 dB theoretical prediction confirms the tiles are
adding coherently rather than by accident.

### 12. A dataset-mutation footgun, found by the test suite

The first cut had the client flip `dataset.return_channels = True` so its batches
would carry channels. But callers reuse that same dataset object to build
evaluation loaders, so `RISClient.evaluate` started receiving 4-tuples and
`test_components.py::test_client` broke. Objectives that need channels now
iterate a `ChannelDatasetView` wrapper instead; the dataset's own `__getitem__`
is unconditionally a 2-tuple. `test_client_does_not_change_dataset_arity` guards
this.

### 13. The classical multi-user optimizer never optimized — *pre-existing*

`_optimize_multiuser_phases` used a fixed `lr = 0.05` against a sum-rate gradient
that scales with absolute received power (~1e-18 at these path losses), so each
step moved the phases by ~1e-6 radians. Measured over 300 iterations the
objective went **0.002081 → 0.002085**, while the random vector it started from
scored 0.002081 and a plain single-user MRC solution scored 0.010603. The
"optimized" phases it returned were indistinguishable from random, which means
**every multi-user number in experiment_10 was a random-phase measurement** —
including the sum-rate and fairness figures the experiment reports as its result.

Fixed by normalizing the gradient by its largest component, so the step is a
fixed angular size (0.5 rad, annealed) and is scale-invariant with respect to
path loss:

| | sum-rate |
|---|---|
| random start | 0.002081 |
| old optimizer (300 iters) | 0.002085 |
| single-user MRC | 0.010603 |
| **fixed optimizer (50 iters)** | **0.015757** |

It now also beats single-user MRC, which is the minimum bar for a multi-user
optimizer to be worth running. Guarded by
`test_multiuser_optimizer_actually_optimizes`.

### Objective comparison, measured at system scale (16 tiles, 1024 elements)

| objective | system SNR | vs no-RIS | vs 1 tile | gap | sum-rate |
|---|---|---|---|---|---|
| `mse`     | −4.61 dB | 38.14 dB | 25.19 dB | **10.58 dB** | 0.890 |
| `snr`     | −6.29 dB | 36.47 dB | 24.35 dB | 12.25 dB | **1.118** |
| `sumrate` | −6.28 dB | 36.47 dB | 24.35 dB | 12.25 dB | **1.118** |

Each objective wins on the metric it optimizes — there is no dominant choice.
`snr` and `sumrate` coincide because at these SINRs the noise term dominates the
cross-talk term, so the interference model barely bites.

### Still open

- The optimality gap at system scale is **10.6 dB** — with the surface now large
  enough to matter, closing this is a model-capacity/training question and is the
  main remaining lever on performance.
- Objective choice is workload-dependent; see the table above. Config ships
  `'sumrate'` because the system is framed as multi-user, but `'mse'` is the
  better default if the headline claim is single-user SNR.
- Duty cycling is deliberately left as-is per the owner's instruction; its
  measured energy/SNR trade-off is documented in `config.py`.
- `target_user = 0` remains in the *label* construction (used only by the MSE
  path). The sum-rate objective and the new all-user evaluation metrics do not
  use it.

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

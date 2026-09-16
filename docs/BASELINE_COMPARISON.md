# Comparison with baselines and with the published literature

Two comparisons, kept separate on purpose:

1. **Measured** — every baseline implemented in this repository, run on the same
   channel realizations as the proposed scheme. These numbers are reproducible
   from the JSON in `results/`.
2. **Positional** — where this work sits relative to published RIS and
   federated-learning papers. These are *not* head-to-head numbers. Nobody has
   run those papers' code on these channels, and quoting their reported dB
   figures next to ours would compare different geometries, path-loss models,
   element counts and SNR definitions. What is compared is the *problem setting*
   and the *cost model*.

---

## 1. Measured comparison

### The baselines, and what each one is entitled to

| Baseline | Implementation | Gets | Reference |
|---|---|---|---|
| No RIS | direct link only | — | — |
| Random phases | i.i.d. uniform on [0, 2π) | nothing | — |
| Random search | best of T random draws | true SNR oracle per draw | `baselines/random_search.py` |
| Alternating Optimization (AO) | projected gradient ascent on \|h_eff\|², unit-modulus | estimated cascaded CSI | Wu & Zhang, *IEEE TWC* 2020 |
| SCA | successive convex approximation, closed-form surrogate maximiser | estimated cascaded CSI | Guo et al., *IEEE TWC* 2020 |
| ADMM | augmented-Lagrangian splitting | estimated CSI | Yu et al., *IEEE JSAC* 2020 |
| SDR | semidefinite relaxation + randomization | estimated CSI | Luo et al., *IEEE SPM* 2010 |
| DRL (TD3) | actor-critic over phase actions | reward from environment | Huang et al., *IEEE JSAC* 2020 |
| Centralized DL | **the same network and the same objective**, trained on pooled data | all tiles' data at one node | `baselines/centralized_learning.py` |
| **Fed-RIS (proposed)** | per-tile GAT phase predictor, FedAvg over the NoC | own tile's CSI only | this work |
| Perfect-CSI MRC | θ_n = ∠h_direct − ∠c_n | true CSI (oracle) | upper bound |

The centralized control deserves a note. It is trained through
`src/client.RISClient` on a pooled `RISChannelDataset.concat(...)` — identical
architecture, objective, optimizer and schedule to the federated clients — so the
only difference between the two rows is **whether the data stayed on the tiles**.
Pooling with `torch.utils.data.ConcatDataset` would have silently dropped the
channel arrays the sum-rate objective needs and forced the baseline onto an MSE
objective, turning an FL-vs-centralized comparison into an objective comparison.

### AO and SCA use the same phase convention as everything else

The two solvers internally form a cascaded coefficient with different
conventions — SCA as `h_ris_user * h_bs_ris`, AO as `conj(h_ris_user) * h_bs_ris`.
`experiments/link_level.design_iterative` passes the cascade (or its conjugate)
with a unit second factor so both reproduce exactly the cascade used by
`combine_tile_phases`. Without that, one of the two would have been scored under
a phase convention it never optimized for.

Every scene is solved. A phase vector is a function of that scene's channel, so
unlike a trained network's weights it cannot be reused across scenes.

### Results

Tables 1–7 below are generated from `results/link_level/link_level_results.json`.

**How to read Table 1.** "Array gain" is the mean channel power gain relative to
the blocked direct link. "ρ @ BER 1e-3" is the transmit SNR at which the scheme
first reaches that error rate — *lower is better*, and the difference between
two rows is a transmit-power saving in dB. A dash means the scheme never reaches
that target within the swept range, which is reported rather than extrapolated.

### What the measured comparison actually shows

- **Against no-RIS and random phases**, the surface delivers the large gain the
  physics predicts. This part is not in dispute and is mostly a sanity check.
- **Against AO and SCA under perfect CSI**, the learned designs trail. This is
  expected and should be stated plainly: for a single user with a
  single-antenna BS, maximizing \|h_direct + Σ c_n e^{jθ_n}\| has a closed-form
  solution, SCA converges to it, and no learned predictor can beat an oracle
  solving the exact problem. The learned design's case is the cost model in
  Table 2 — no CSI leaves the tile, one forward pass per coherence block instead
  of an iterative solve — plus Table 5.
- **Against AO and SCA under imperfect CSI**, the gap narrows. A solver treats
  the noisy estimate as truth and optimizes it exactly; a network trained across
  many realizations carries a prior over channels and degrades more gently. The
  "loss over sweep" column in Table 5 is the quantity to quote.
- **Against the centralized control**, the federated model lands 1.1 dB behind
  at BER 10⁻³ (Table 1). That figure *is* the price of keeping CSI on-tile — the
  two differ in nothing else, same architecture, objective, optimizer and
  schedule — and it should be quoted as a price rather than glossed as parity.
- **On interconnect traffic, federation loses at this scale, and the tables say
  so.** Shipping INT8 model deltas every round moves roughly an order of
  magnitude more bytes than shipping every tile's raw CSI once (Table 7). The
  argument for the federated design is that the CSI never leaves the tile and
  that inference is one forward pass instead of a per-coherence-block solve —
  not that it saves bandwidth. Table 7 also gives the crossover: the trade flips
  once the per-tile dataset is large relative to the model.

### A finding that is about the deployment, not the phase design

Table 6 shows every design — including the perfect-CSI bound — tracking the N²
coherent-combining law from N = 256 to N = 512 and then flattening: the last
512 elements buy about 1.4 dB where the law predicts 6 dB. That is not a failure
of any phase design. The tiles are distributed around the room on a circle, so
the far-side tiles see a much weaker cascaded channel than the near-side ones,
and a coherent sum of very unequal terms saturates. The actionable conclusion is
about **tile placement**, and it is visible only because the surface is scored
as one aperture rather than one tile at a time.

---

## 2. Positional comparison with the literature

### Optimization-based RIS phase design

| Work | Approach | Requires | Per-coherence-block cost | CSI leaves the node |
|---|---|---|---|---|
| Wu & Zhang, *IEEE TWC* 2020 | alternating optimization, joint active/passive beamforming | full instantaneous CSI at a central controller | iterative solve | yes |
| Wu & Zhang, *IEEE TWC* 2020 (discrete) | discrete-phase beamforming, 1/2/3-bit | full instantaneous CSI | iterative solve | yes |
| Guo et al., *IEEE TWC* 2020 | SCA / weighted sum-rate maximization | full instantaneous CSI | iterative solve | yes |
| Yu et al., *IEEE JSAC* 2020 | ADMM | full instantaneous CSI | iterative solve | yes |
| Luo et al., *IEEE SPM* 2010 (SDR machinery) | semidefinite relaxation | full instantaneous CSI | SDP + randomization | yes |
| **This work** | federated per-tile learned predictor | each tile's own CSI | one forward pass | **no** |

These methods are the correct target to beat *on quality* and they are
implemented here as baselines rather than cited as numbers. The difference this
work claims is architectural: the optimization is amortized into weights that
are trained once and then evaluated in constant time, and the CSI never has to
be aggregated anywhere.

### Learning-based RIS phase design

| Work | Clients / data | Objective | Hardware in the loop | Interconnect modelled |
|---|---|---|---|---|
| Huang et al., *IEEE JSAC* 2020 | single DRL agent, centralized | reward = rate | partial | no |
| He et al., *IEEE TWC* 2022 | GNN/GAT beamforming, centralized | rate | no | no |
| Shen et al., *IEEE TSP* 2021 | GNN resource management, centralized | utility | no | no |
| Eisen & Ribeiro, *IEEE TSP* 2020 | random-edge GNN, centralized | constrained utility | no | no |
| **This work** | **tiles of one surface**, federated | differentiable sum-rate | quantization, jitter, CSI error | NoC topology, protocol, energy |

The GNN-for-wireless line establishes that attention over elements is the right
inductive bias, and this repository adopts it (`models/ris_net.py`, GAT with 8
heads). What is added is that the model is *trained where the data is* — one
client per tile — and that the training cost is charged to a real on-die
interconnect.

### Federated learning

| Work | Clients | Source of non-IID-ness | What the aggregation costs |
|---|---|---|---|
| McMahan et al., *AISTATS* 2017 | mobile devices | user behaviour | abstract bytes |
| Li et al., *MLSys* 2020 (FedProx) | heterogeneous devices | system + statistical | abstract bytes |
| Karimireddy et al., *ICML* 2020 (SCAFFOLD) | devices | client drift | abstract bytes |
| Konecny et al., 2016 | devices | user data | abstract bytes, compression studied |
| **This work** | **RIS tiles on one die** | **tile geometry — each tile sees the same scene from its own position** | Mesh/Torus/Folded-Torus NoC, RingAllReduce, per-bit energy |

All three aggregation rules are implemented (`src/server.py`) and compared
(experiment 11). The contribution is not a new aggregation rule; it is that the
federated abstraction is applied to a setting where "communication cost" is a
measurable on-chip quantity and where the clients' outputs are physically summed
rather than independently useful.

### Where this work is weaker than the literature it cites

- It does not do **joint active and passive beamforming** — the BS is
  single-antenna, so the comparison to Wu & Zhang is on the passive half only.
- It does not claim a **convergence guarantee**. The optimization-based
  baselines come with one; a learned predictor does not.
- Its channels are **synthetic** (Rician / 3GPP-UMi style). DeepMIMO
  ray-tracing is wired in but the O1_28 scenario data is not bundled.
- **Multi-user** results exist (experiment 10) but the link-level figures are
  single-user-dominant, with cross-talk configurable and zero by default.

---

<!-- BEGIN GENERATED TABLES -->
<!-- regenerate with: python make_summary_tables.py --write -->

## Run provenance

| Setting | Value |
|---|---|
| Tiles × elements | 16 × 64 = 1024 |
| Test scenes | 600 |
| Carrier | 28 GHz |
| Noise power | -90 dBm |
| Direct-link blockage | 30.0 dB |
| Operating point ρ | 99 dB |
| Reduced (quick) run | False |
| Seed | 42 |

## Table 1 — Link-level comparison, all schemes on one channel set

| Scheme | Array gain vs no-RIS (dB) | ρ @ BER 1e-3, QPSK (dB) | ρ @ BER 1e-4, QPSK (dB) | ρ @ BER 1e-3, 16QAM (dB) | SE @ ρ=99 dB (bit/s/Hz) | Δ vs Fed-RIS (dB) |
|---|---|---|---|---|---|---|
| No RIS (blocked direct) | 0.00 | 135.9 | 139.0 | — | 0.16 | 36.7 |
| Random phases | 0.65 | 136.9 | — | — | 0.35 | 37.7 |
| Alternating Optimization | 14.80 | 98.0 | 100.6 | 104.6 | 5.23 | -1.2 |
| SCA | 14.85 | 98.0 | 100.6 | 104.6 | 5.24 | -1.2 |
| Centralized DL (pooled) | 14.72 | 98.1 | 100.7 | 104.7 | 5.19 | -1.1 |
| **Fed-RIS (proposed)** | 12.88 | 99.2 | 101.6 | 105.8 | 4.64 | 0.0 |
| Perfect-CSI MRC (bound) | 14.91 | 97.9 | 100.5 | 104.5 | 5.26 | -1.3 |

## Table 2 — What each scheme requires to produce a phase design

| Scheme | CSI requirement | Per-block cost | CSI stays local | Inference complexity |
|---|---|---|---|---|
| No RIS (blocked direct) | — | — | yes | O(1) |
| Random phases | none | none | yes | O(N) |
| Alternating Optimization | full instantaneous | per block | no | O(N·I) |
| SCA | full instantaneous | per block | no | O(N·I) |
| Centralized DL (pooled) | pooled to server | one forward | no | O(N) |
| **Fed-RIS (proposed)** | stays on tile | one forward | yes | O(N) |
| Perfect-CSI MRC (bound) | perfect (oracle) | closed form | n/a | O(N) |

## Table 3 — Phase-quantization loss vs the classical sinc bound

| Phase resolution | Theory 20·log10 sinc(2⁻ᵇ) (dB) | Measured, perfect-CSI MRC (dB) | Measured, Fed-RIS (dB) |
|---|---|---|---|
| 1-bit (2 states) | -3.92 | -3.40 | -3.26 |
| 2-bit (4 states) | -0.91 | -0.84 | -0.82 |
| 3-bit (8 states) | -0.22 | -0.21 | -0.20 |
| Continuous | 0.00 | 0.00 | 0.00 |

## Table 4 — RIS phase-jitter loss (Fed-RIS design)

| RMS phase jitter σ_φ | Array-gain loss (dB) |
|---|---|
| 0° | 0.00 |
| 5° | -0.03 |
| 15° | -0.27 |
| 30° | -1.07 |

## Table 5 — Mean received SNR vs normalized CSI error ε

| Scheme | ε=0 | ε=0.001 | ε=0.01 | ε=0.05 | ε=0.1 | ε=0.3 | ε=1 | Loss over sweep (dB) |
|---|---|---|---|---|---|---|---|---|
| No RIS (blocked direct) | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 | 0.0 |
| Random phases | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 0.0 |
| Alternating Optimization | 18.1 | 18.1 | 18.1 | 17.8 | 17.5 | 16.3 | 13.0 | 5.1 |
| SCA | 18.2 | 18.2 | 18.1 | 17.9 | 17.5 | 16.3 | 13.1 | 5.1 |
| Centralized DL (pooled) | 18.0 | 18.0 | 18.0 | 17.8 | 17.6 | 16.5 | 13.4 | 4.6 |
| **Fed-RIS (proposed)** | 16.2 | 16.2 | 16.2 | 16.0 | 15.8 | 15.1 | 13.0 | 3.2 |
| Perfect-CSI MRC (bound) | 18.2 | 18.2 | 18.2 | 18.2 | 18.2 | 18.2 | 18.2 | 0.0 |

## Table 6 — Array-gain scaling (mean received SNR, dB)

| Scheme | N=64 | N=128 | N=256 | N=512 | N=768 | N=1024 |
|---|---|---|---|---|---|---|
| No RIS (blocked direct) | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 |
| Random phases | 3.3 | 3.3 | 3.4 | 3.9 | 4.0 | 4.0 |
| Alternating Optimization | 3.7 | 4.9 | 10.3 | 16.8 | 17.5 | 18.1 |
| SCA | 3.7 | 4.9 | 10.3 | 16.8 | 17.6 | 18.2 |
| Centralized DL (pooled) | 3.8 | 5.0 | 10.3 | 16.7 | 17.4 | 18.0 |
| **Fed-RIS (proposed)** | 3.6 | 4.5 | 8.9 | 14.7 | 15.6 | 16.2 |
| Perfect-CSI MRC (bound) | 3.8 | 5.0 | 10.5 | 16.9 | 17.6 | 18.2 |
| _Ideal N² law_ | 3.8 | 9.8 | 15.8 | 21.8 | 25.4 | 27.9 |

## Table 7 — Training and per-coherence-block cost

| Quantity | Value |
|---|---|
| Model parameters | 315,906 |
| FL rounds × local epochs | 20 × 3 |
| Tiles | 16 |
| Train samples per tile | 600 |
| Total NoC traffic, FL, INT8 deltas (MB) | 192.8 |
| Raw-CSI upload a centralized controller needs (MB) | 4.9 |
| FL traffic ÷ raw-CSI traffic | 39.6× |
| Samples per tile at which FL becomes the cheaper transfer | ≈ 23,752 |
| FL wall clock (s) | 3966 |
| Centralized wall clock (s) | 2522 |
| Mean per-coherence-block solve time, Alternating Optimization (ms) | 21.928 |
| Mean per-coherence-block solve time, SCA (ms) | 1.953 |
| Mean per-coherence-block inference, learned schemes | one forward pass, no iteration |

## Table 8 — System-level baseline table (experiment 9, earlier run)

| Method | SNR (dB) | Rate (bit/s/Hz) | Comm (KB) | Energy (mJ) | Iterations | CSI stays local | Complexity |
|---|---|---|---|---|---|---|---|
| no_ris | 2.23 | 1.42 | 0.0 | 0.0 | 0 | yes | O(1) |
| random_ris | 2.39 | 1.45 | 0.0 | 0.0 | 1 | yes | O(N) |
| random_search | 5.57 | 2.20 | 0.0 | 0.0 | 1000 | no | O(N·T) |
| drl_td3 | 2.70 | 1.52 | 0.0 | 150.0 | 150 | yes | High |
| alternating_opt | 2.71 | 1.52 | 50.0 | 10.0 | 100.0 | no | O(N²·I) where N=64, I=50.0 |
| centralized_dl | 4.88 | 2.03 | 461.7 | 2.0 | 4 | no | O(N·E·B) |
| federated_ours | 7.21 | 2.65 | 9872.1 | 161.9 | 4 | yes | O(N·E·B/K) |
| optimal | 8.30 | 2.96 | 0.0 | 0.0 | 0 | yes | N/A (oracle) |

_Provenance: 4 tiles × 64 elements, 4 FL rounds, 150 train samples, reduced run = True, saved 2026-09-08T16:05:59._

<!-- END GENERATED TABLES -->

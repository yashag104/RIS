# Pilot-Based Estimation and Equalisation of Harmonic-Coupled OFDM Channels through Dispersive Time-Modulated RIS

MATLAB R2026a, base MATLAB only (no Communications Toolbox). Independent of the
Python FL/RIS codebase in the parent folder.

Literature map and gap: `RESEARCH_NOTES.md`.

## Status (2026-10-01)

| Gate | What | Result |
|---|---|---|
| 0 | Novelty: nobody equalises or estimates TM-RIS harmonic coupling at a legitimate receiver | Passed (searched; see notes) |
| 1 | Reproduce Iudice et al. (arXiv 2609.18360) and validate the closed form against an independent time-domain simulation | **Passed** |
| 2 | Joint MMSE vs treat-interference-as-noise (TIN), perfect H; pre-registered energy-collection prediction | **Passed** |
| 3 | Structured pilot estimator within ~1 dB of perfect H with Np <= 2 | **Passed** (physics-aware LMMSE + decision-directed); one marginal miss in the non-physical channel (see "Gate 3, closed") |

### Gate 1 (`matlab/gate1_reproduce.m`, `results/gate1.json`)
* Dispersion of the OpenRIS cell over 98 MHz. Differential phase 172–184 deg (paper:
  180 +/- 6.5), imbalance +/-0.57 dB (< 0.75), per-state phase variation 46.5/49.2 deg
  (~47/50), ripple 0.62/0.50 dB (~0.6). Matches.
* K = 2 coupling: odd harmonics on the sinc(h/2) envelope; even harmonics exactly zero;
  the DC harmonic appears only through dispersion, at -19 to -27 dB relative to h = 1.
* Closed-form coupling matrix vs direct time-domain simulation (filter -> gate -> FFT,
  Q = 4 elements, K = 4 random sequences, multipath on both hops): relative error
  0.27% median, 0.49% max (residual of sampled vs continuous gates).

### Gate 2 (`matlab/gate2_equalization.m`, `results/gate2.json`)
Q = 16 elements, K = 8 slots, 256 subcarriers (central 128 scored), 30 trials,
dispersive cell, 30 ns RMS multipath per hop.

| Scenario | duty D | energy-collection gain, measured | predicted (ideal) | rate @ 20 dB: TIN / MMSE / MF bound |
|---|---|---|---|---|
| S1 same sequence | 0.5   | 23.4 dB | inf (b0 = 0) | 0.01 / 6.22 / 6.54 |
| S1 | 0.625 | 11.6 | 12.0 | 0.10 / 6.14 / 6.45 |
| S1 | 0.75  | 5.9  | 6.0  | 0.42 / 6.19 / 6.52 |
| S1 | 0.875 | 2.4  | 2.5  | 1.19 / 6.11 / 6.43 |
| S2 random shifts | 0.5 | 26.4 | – | 0.01 / 5.59 / 6.61 |
| S2 | 0.75 | 6.5 | – | 0.62 / 5.13 / 6.62 |

* The measured gain matches the closed-form prediction 1/(2D-1)^2 to within 0.1–0.4 dB.
* MMSE tracks the MF bound within ~0.3 bit/s/Hz for S1. For random space-time shifts
  (S2) the gap grows to ~1.5 bit/s/Hz, because H is worse conditioned. That motivates
  sequence design for conditioning (week 3).
* With binary 50% duty (the paper's own example), TIN gets essentially nothing.
  Joint equalisation recovers about 95% of the MF-bound rate.

Caveat for the paper: the TIN gap depends on how much ICI reaches the user. A user in
a TM-IRS beam *designed* to keep h = 0 dominant will see a smaller gain than these
generic-position numbers. This must be shown explicitly, not hidden.

### Gate 3, first attempt: FAILED (`matlab/gate3_estimation.m`, `results/gate3.json`)
Model-agnostic estimator H = sum_h Shift_h diag(F c_h) with +/-8 or +/-12 harmonics and
delay-limited gains. NMSE floors at -14 to -16 dB whatever the pilot count, and the
rate after MMSE is only 2–3 bit/s/Hz against 5–6 with perfect H. Cause: rectangular
switching leaves a sinc^2 ~ 1/h^2 harmonic tail, so ~3–5% of the energy lies outside
any small harmonic set. Baselines at 20 dB: conventional one-tap 0.4, Toeplitz ~1.4–2.6,
banded unstructured LS ~0–0.6 bit/s/Hz.

### Gate 3, physics-aware estimator: PASSED for SNR <= 20 dB (`gate3b_physics.m`, `gate3c_tune.m`)
The legitimate receiver knows the control sequence and the element-response family,
so all harmonics are tied by a known matrix B:
S1: H = B o (Fd C Fs^T), with C the Ld x Ls two-hop delay matrix (64 unknowns).
Truth keeps all in-band harmonics.

| SNR | Np = 1 | Np = 2 | Np = 4 | perfect H |
|---|---|---|---|---|
| 10 dB | 3.21 | 3.21 | 3.26 | 3.31 |
| 20 dB | 6.21 | 6.30 | 6.23* | 6.45 |
| 30 dB | 7.9 | 8.3–8.6 | 9.56* | 9.6–9.8 |

(rate after MMSE, bit/s/Hz. Rows 20/30 dB at Np = 1, 2 use the tuned ridge (osr 2,
relative ridge 1e-3..1e-2); * = gate3b, untuned ridge.)

* Pre-registered criterion (Np <= 2, within ~1 dB of perfect) is met at 10 and 20 dB,
  but NOT at 30 dB, where 4 pilot symbols are needed. Residual NMSE floor ~ -30 dB from
  off-grid delays in the delay basis (DPSS basis is the planned fix).
* **Dispersion-aware B matters.** B from the ideal +/-1 model floors at -24.5 dB NMSE.
  The exact model reaches -52 dB. At 30 dB / Np = 4: 9.56 vs 7.34 bit/s/Hz.
* S2 (per-element random shifts, independent per-element channels): 1024 unknowns, so it
  needs Np >= 8 (5.0 vs 5.1 bit/s/Hz at 20 dB). The independent-per-element channel is
  Iudice's general form but unphysical for a compact surface. Co-located elements share
  path delays, which should cut the unknowns to ~64 + Q phases (week 3).

### Gate 3, closed (`gate3d_fix.m`, `gate3d_lmmse.m`, `gate3e_dd.m`)
One calibrated setting (DPSS basis, tau_max = 0.3 us, L = 16, LMMSE with DPSS-eigenvalue
prior), chosen on a calibration seed and evaluated on fresh seeds. Per-element sequences
use a rank-1 (a_q s_q^T) ALS estimator. Channels: Iudice's general independent
per-element TDL, and a physically consistent model (shared paths, geometric phases,
K = 10 dB).

Loss vs perfect-H MMSE, dB, Np = 2 pilot symbols (rate-equivalent):

| case | 10 dB | 20 dB | 30 dB pilots only | 30 dB + decision-directed (2 iters, 6 data symbols) |
|---|---|---|---|---|
| iid S1 | 0.06 | 0.45 | 1.06 | 0.3 |
| iid G4 | 0.29 | 0.58 | 1.87 | **1.2** |
| iid S2 | 0.30 | 0.56 | 1.31 | 0.7 |
| phys S1 | 0.04 | 0.16 | 0.47 | – |
| phys G4 | 0.29 | 0.57 | 1.17 | 0.5 |
| phys S2 | 0.38 | 0.56 | 1.19 | 0.5 |

Verdict: criterion met everywhere for the physical channel. The single miss is iid G4 at 30 dB
(1.2 dB, limited by 64-QAM decision errors at SER 2–4%) under the non-physical
independent-per-element channel. Reported as is.

## Week 3 results

### (a) Fair case: TM designed for the legitimate user (`week3_designed_user.m`)
Per-element polarity (sequence or its complement) co-phases h = 0 at the user;
random per-element shifts keep the signal scrambled elsewhere. Physical channel, Np = 2 estimate.

| duty | user | one-tap SIR | TIN rate 10/20/30 dB | joint MMSE (estimated) 10/20/30 dB |
|---|---|---|---|---|
| 0.875 | designed | +10.0 dB | 2.51 / 3.34 / 3.47 | 3.13 / 6.18 / 9.37 |
| 0.75  | designed | +3.5 dB | 1.44 / 1.70 / 1.74 | 2.77 / 5.63 / 8.80 |
| 0.625 | designed | -2.3 dB | 0.62 / 0.70 / 0.71 | 2.24 / 4.53 / 7.38 |
| 0.75  | generic  | -12.3 dB | 0.17 / 0.19 / 0.19 | 2.24 / 4.77 / 7.66 |

Even in the most TIN-friendly configuration, TIN saturates at its SIR (~3.5 bit/s/Hz).
The gain is modest at 10 dB (+0.6 bit/s/Hz, ~2 dB) and large at 20–30 dB.
Report it that way.

### (b) Uncoded BER, 16-QAM (`week3_banded_ber.m`)
* One-tap: BER 0.39–0.45, unusable without joint equalisation.
* Full MMSE with the estimated channel (Np = 2 + 2 DD iterations) matches perfect-H BER
  (S1: 2.9e-3 at 16 dB, 2e-5 at 20 dB, 0 at 24 dB).
* **Banded sliding-window MMSE fails.** With W = 16 it floors at 2.5e-3 (S1) / 3e-2 (S2): the
  1/h^2 harmonic tail reaches far outside any short window. Negative result, worth stating.
* S2 (random shifts) has a high BER even with perfect H (7.7e-3 at 28 dB), because of conditioning.

### (c) Exact fast operator + CG equaliser (`fast_coupling.m`, `week3_fast_eq.m`)
* Key identity: B_q(mbar, m) = sum_s T_{q,s}(mbar - m) Gamma_s(f_m). Each element's coupling
  is an exact sum of 2 Toeplitz matrices times the dispersive state responses, so H x and
  H' y cost O(Q M log M) by FFT, with no M x M matrix needed.
* Verified against the explicit matrix: relative error 7e-16.
* 20 CG iterations give the same SER as the direct full MMSE (perfect and estimated H).
* Honest timing: in MATLAB the dense direct solve is still faster in wall-clock up to
  M = 1024 (0.15 s vs 0.54 s). The advantage is asymptotic in operations and memory
  (M = 3276 dense H is 170 MB). The paper must state flops/memory, not "faster".

### (d) Channel-aware shift design (`week3_seq_design.m`), INDICATIVE (8 trials, 20 dB)
Greedy per-element shift choice that maximises the legit user's MMSE rate:
5.58 -> 8.85 bit/s/Hz (identical shifts: 5.52). The eavesdropper's one-tap SIR stays scrambled
(-6.5 -> -8.2 dB). Most of the gain is harmonic *beamforming* (shifts set harmonic phases,
so power adds coherently at the user, at fixed Tx power and noise). That is the known
space-time-coding harmonic-steering idea (Zhang/Cui 2018). Only the joint MMSE-rate /
dispersive design is new. Supporting result, not a headline.

### (e) Element-model mismatch, and a model-free estimator (`week3_mismatch.m`, `week3_modelfree.m`)
Truth uses Table I. The receiver's B uses shifted resonances (df0) and +/-10% decay rates.
Np = 2 + 2 DD iterations. Loss vs perfect H, dB at 20 / 30 dB:

| receiver knowledge | S1 | S2 (rank-1) |
|---|---|---|
| exact model | 0.05 / 0.12 | 0.24 / 0.51 |
| df0 = 5 MHz | 0.53 / 2.59 | 0.24 / 0.49 |
| df0 = 10 MHz | 0.66 / 3.36 | 0.24 / 0.50 |
| df0 = 20 MHz | 0.55 / 2.91 | 0.24 / 0.51 |
| ideal +/-1 (frequency-flat literature model) | 1.41 / 6.57 | 0.24 / 0.53 |
| **model-free per-state (timing only)** | **0.08 / 0.11** | not run yet |

* The physics-aware S1 estimator passes the stated criterion (< 1 dB at 20 dB for df0 <= 10 MHz)
  but is fragile at 30 dB.
* **Model-free estimator.** B_q(mbar, m) = sum_s T_{q,s}(mbar - m) Gamma_s(f_m), and T_{q,s}
  depends only on the control timing. Gamma_s(f_m) multiplies the input side, so it is absorbed
  into a per-state input response: H = sum_s T_s o (F C_s F^T), with 2 L^2 unknowns.
  It needs NO element model, matches the exact-model estimator (0.08 / 0.11 dB), and
  removes the 3–6.5 dB loss of mismatched or frequency-flat models. This replaces
  "dispersion-aware B" as the headline estimator.
* S2 under mismatch is insensitive (losses dominated by estimation noise). A per-(element, state)
  model-free variant for S2 still needs a bilinear form (shared a_q, per-state s_{q,s}); to do.

### (f) Full n78 band, M = 3276, matrix-free (`week3_fullband*.m`)
Estimation regressors and equalisation use only the exact FFT operator (no M x M matrix).
* First run (full delay matrix C, L = 36). SER at 20 dB / 28 dB (S1): estimated 0.045 / 0.057
  vs true-H 0.033 / 0.018. Gap at high SNR. More DD passes made it WORSE (0.24):
  64-QAM decision errors propagate.
* Diagnosis (`week3_fullband_err.m`). CG converges by 25 iterations, so it is not the cause. The model is correct:
  operator NMSE reaches -45..-49 dB at 60 dB SNR. At 28 dB the full-C estimator is
  noise-limited at -23..-29 dB, because the two-hop delay matrix has ~(2NW)^2 effective unknowns,
  which grows quadratically with bandwidth.
* Fix: C is low rank (shared propagation paths: rank <= #paths). A rank-r ALS estimator
  C = U V^T has 2 L r unknowns: -39.0 dB at Np = 2 (rank 6 or 16, same), vs -26.3 dB for full C.
* SER with low rank (S1, L = 30):

| SNR / QAM | one-tap | CG true H | estimate, pilots only | estimate + 1 DD |
|---|---|---|---|---|
| 20 dB / 16-QAM | 0.84 | 0.0327 | 0.034 | 0.034 |
| 28 dB / 64-QAM | 0.96 | 0.0182 | 0.042 | 0.026 |

* Open: S2 (per-element sequences) at full band is still far from true H
  (28 dB: 0.16–0.18 vs 0.074–0.081). Planned fix: per-element factors on a SHARED low-rank
  path subspace (a_q = U alpha_q, s_q = V beta_q) instead of independent rank-1 per element.

## Run
```
matlab -batch "cd('C:\Users\hp\RIS\tm_ris_ofdm\matlab'); gate1_reproduce"
matlab -batch "cd('C:\Users\hp\RIS\tm_ris_ofdm\matlab'); gate2_equalization"
```

## Plan
* Week 2: done (Gate 3 above).
* Week 3: done (results above).
* Remaining before writing: full-band scale (M = 3276) run using the fast operator;
  more trials for (d); robustness to element-model mismatch (receiver's Table I
  parameters perturbed by +/-5%, since the physics-aware estimator depends on them);
  coded BER (optional).
* Weeks 4–5: letter draft.

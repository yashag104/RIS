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
| 3 | Structured pilot estimator beats per-subcarrier / Toeplitz baselines at equal pilots | **Passed, with the physics-aware estimator, for SNR <= 20 dB** (details below); the first, model-agnostic estimator failed |

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

## Run
```
matlab -batch "cd('C:\Users\hp\RIS\tm_ris_ofdm\matlab'); gate1_reproduce"
matlab -batch "cd('C:\Users\hp\RIS\tm_ris_ofdm\matlab'); gate2_equalization"
```

## Plan
* Week 2: done (Gate 3 above).
* Week 3: (a) physically consistent channel (shared path delays, per-element geometry
  phases) and the S2 estimator on it; (b) DPSS delay basis for the 30 dB floor;
  (c) designed-beam user, the fair case for TIN; (d) sequence choice for conditioning;
  (e) low-complexity banded equaliser and BER Monte Carlo.
* Weeks 4–5: letter draft.

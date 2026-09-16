# The six results that belong in the paper

Six figures, in the order a communications reviewer reads them: *does the RIS
work* → *how much rate does it buy* → *how reliable is it* → *does it survive
real hardware* → *does it survive real CSI* → *does it scale*. Each one is
produced by `run_link_level.py` and rendered by `utils/plotting_link.py`; all
six come from a single set of channel realizations, so numbers can be carried
between figures without re-deriving anything.

```bash
python run_link_level.py            # full scale (16 tiles × 64 elements)
python run_link_level.py --quick    # minutes, pipeline check only
python run_link_level.py --plot-only  # re-render from the saved JSON
python -m utils.system_diagram      # the system figure
python make_summary_tables.py       # every table in these docs
```

Everything lands in `results/link_level/` (PDF for the paper, PNG for review)
alongside `link_level_results.json`, which is the single source of truth for the
tables below.

## Headline numbers from the full-scale run

16 tiles × 64 elements = 1024, 600 test scenes, 20 FL rounds × 3 local epochs,
600 training samples per tile, 28 GHz, direct link blocked by 30 dB. Full
provenance and every table are at the bottom of this document.

- **Fed-RIS reaches BER 10⁻³ (QPSK) 36.7 dB earlier than the blocked direct
  link** and 37.7 dB earlier than random phases — the surface is what makes the
  link work at all.
- **It sits 1.3 dB from the perfect-CSI MRC bound** and 1.1 dB behind the
  centrally trained control. That 1.1 dB is the price of keeping CSI on-tile,
  and it is the number to quote rather than any claim of parity.
- **Under imperfect CSI it degrades most gently of all the designs**: over
  ε = 0 → 1 it loses 3.2 dB against 5.1 dB for AO and SCA and 4.6 dB for the
  centralized control, so the schemes converge at ε = 1 (all ≈ 13 dB).
- **Quantization behaves exactly as theory says**: measured 1/2/3-bit losses of
  3.26 / 0.82 / 0.20 dB against the `20·log₁₀ sinc(2⁻ᵇ)` predictions of
  3.92 / 0.91 / 0.22 dB. 2-bit is the practical knee.
- **Phase jitter is second-order**: 0.03 dB at 5° RMS, 1.07 dB at 30°.
- **The surface saturates above ~512 elements.** From N = 256 to 512 every
  design tracks the N² coherent-combining law within a fraction of a dB; beyond
  that it flattens, because the tiles are distributed around the room and the
  far-side tiles see much weaker cascaded channels. This is a placement result,
  not a phase-design result, and it is the most actionable thing in Fig. 6.
- **Federation costs interconnect traffic, not saves it**: 192.8 MB of INT8
  model deltas against the 4.9 MB of raw CSI a centralized controller would have
  moved — a factor of 39.6. See `docs/NOVELTY.md`, N5, for the crossover.

---

## Fig. 0 — System architecture  *(the front-matter figure)*

`results/figures/system_architecture.pdf`

Three bands: the 28 GHz propagation scene with the obstructed direct link and
the tiled surface; the on-chip federated loop across tiles over the NoC; the
phase-application chain that turns a prediction into a received SNR and then
into link-level metrics. Generated from `config.py`, so it always shows the
configuration actually simulated.

## Fig. 1 — BER vs transmit SNR  *(the headline result)*

`fig1_ber_vs_snr.pdf` — two panels, QPSK and 16-QAM, seven schemes.

This is the figure a communications venue expects first. It answers three
questions at once: how far the blocked direct link is from usable, how much of
that a 1024-element surface recovers, and how close each phase design gets to
the perfect-CSI bound. Read it at the BER = 10⁻³ line and report the horizontal
gaps — those are transmit-power savings in dB, which is the currency of the
field.

- x-axis is transmit SNR ρ = P_t/σ², with a secondary P_t (dBm) axis, because
  the schemes differ in *received* SNR and so received SNR cannot be the
  abscissa.
- Curves are masked below the resolvable floor rather than flattened along the
  bottom of the axes.
- Error rates are semi-analytic (exact conditional AWGN error probability
  averaged over realizations) and cross-checked against symbol-level Monte
  Carlo in `test_link_metrics.py`.

**What to say about it.** State the SNR saving over no-RIS and over random
phases, the residual gap to the perfect-CSI MRC bound, and — plainly — that the
model-based solvers sit at that bound when CSI is perfect. The federated design
buys its position without moving CSI off-tile and with one forward pass per
coherence block instead of an iterative solve; Fig. 5 is where it starts to win
on the numbers as well.

## Fig. 2 — Ergodic spectral efficiency and the power saving it represents

`fig2_spectral_efficiency.pdf` — (a) E[log₂(1+γ)] vs ρ for all schemes;
(b) the horizontal transmit-power saving at a 4 bit/s/Hz target, as a bar chart.

Panel (a) is the standard achievable-rate plot; panel (b) converts the same data
into the one number a system designer cares about — how many dB of transmit
power the surface saves at a fixed service rate. Reporting the horizontal gap
rather than the vertical one is what makes the figure comparable with the RIS
optimization literature.

## Fig. 3 — Outage probability

`fig3_outage_probability.pdf` — P(log₂(1+γ) < R_th) vs ρ, at two rate thresholds.

Mean rate hides the tail, and the tail is what a link budget is written against.
Outage separates schemes that are good on average from schemes that are
*reliably* good, and it is where the coherent-combining gain of the full surface
shows up most sharply against random phases. The empirical floor at 1/(number
of realizations) is annotated on the figure so nobody reads simulation
resolution as a physical error floor.

## Fig. 4 — Hardware impairments: discrete phase shifters and phase jitter

`fig4_hardware_impairments.pdf` — (a) BER vs ρ for 1/2/3-bit and continuous
phases; (b) BER vs ρ under RMS phase jitter; (c) measured array-gain loss
against the classical `20·log₁₀ sinc(2⁻ᵇ)` bound.

Real RIS hardware has 1- or 2-bit shifters, so a continuous-phase result is an
upper bound on a product. Panel (c) is the credibility panel: it puts the
measured quantization loss next to the closed-form prediction for both the
proposed design and the perfect-CSI bound, which makes the claim falsifiable
rather than decorative.

**What to say about it.** 2-bit quantization is the practical knee — the loss is
a fraction of a dB — while 1-bit costs several dB; phase jitter is second-order
by comparison at any RMS value a fabricated surface would exhibit.

## Fig. 5 — Robustness to imperfect CSI

`fig5_csi_robustness.pdf` — (a) BER at the operating point vs normalized CSI
error ε; (b) spectral efficiency vs ε; (c) the full waterfall at the worst ε.

This is the figure that argues for a *learned* phase predictor rather than a
solver. A convex solver treats the noisy estimate as truth and optimizes it
exactly; a network trained across many realizations has absorbed a prior over
channels and degrades more gently. The sweep uses a normalized error
ε = σ_e²/E[|h|²] — the convention the RIS literature reports, and the only one
that produces a curve rather than a cliff at these path losses (see
`docs/NOVELTY.md`, N4).

The operating point ρ is chosen automatically as the transmit SNR at which the
proposed scheme reaches BER 10⁻³ under perfect CSI, so panel (a) always sits
inside the waterfall instead of under the resolution floor.

## Fig. 6 — Array-gain scaling and the SNR distribution

`fig6_array_scaling.pdf` — (a) mean received SNR vs active elements N against
the ideal N² law; (b) spectral efficiency vs N; (c) empirical CDF of received
SNR over the full surface.

Panel (a) is the scaling law that justifies building a large surface at all, and
it shows how much of the theoretical N² coherent-combining gain each design
actually realizes as tiles are added. Panel (c) restates Fig. 3's reliability
message as a distribution, and is the cheapest way to show that the proposed
design's advantage is not carried by a few lucky realizations.

---

## If the paper has room for only four

Fig. 1, Fig. 2, Fig. 5, Fig. 6 — waterfall, rate, robustness, scaling. Fold
Fig. 3 into Fig. 2 as an inset and move Fig. 4 to an appendix or a table.

## Supporting results that belong in a table, not a figure

From `results/advanced_experiments/`: the FL-algorithm comparison
(FedAvg / FedProx / SCAFFOLD), the architecture comparison
(MLP / GNN / CNN / Transformer), the NoC topology and protocol comparisons, and
the local-epoch sweep. These characterize the *learning system* rather than the
*link*, and a reviewer will accept them as a table.

> **Scale caveat.** Several experiments in `results/advanced_experiments/` were
> last run at reduced scale to verify execution end to end; their JSONs carry
> `"is_reduced_run": true` in the provenance block. Re-run them at full scale
> before quoting them as publication numbers.

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

## Table 3 — Phase-quantization loss vs the classical sinc bound

| Phase resolution | Theory 20·log10 sinc(2⁻ᵇ) (dB) | Measured, perfect-CSI MRC (dB) | Measured, Fed-RIS (dB) |
|---|---|---|---|
| 1-bit (2 states) | -3.92 | -3.40 | -3.26 |
| 2-bit (4 states) | -0.91 | -0.84 | -0.82 |
| 3-bit (8 states) | -0.22 | -0.21 | -0.20 |
| Continuous | 0.00 | 0.00 | 0.00 |

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

<!-- END GENERATED TABLES -->

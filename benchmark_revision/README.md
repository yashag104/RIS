# Benchmark paper revision: Day-1 gate checks

**Status (2026-09-29): gate FAILED. The paper's central story does not survive, and the corrected story is already in the literature. Do not start the 2–3 week revision.**

Script: `gate_checks.py` (seeds 42 and 123, same generator, SNR and scoring as
`run_pilot_limited.py`). Raw output: `results/gate_checks.json`.

## A. Theory check: PASSED

Full-probe LS matches the closed-form Rician prediction almost exactly: simulated 0.315/0.320 of oracle power vs theory 0.314/0.320. The simulator and the scoring are correct.

## B. LMMSE: a BUG CONFIRMED, and Table II is wrong

| M | paper LMMSE (600 noisy labels) | covariance from 5000 true channels | same, orthogonal DFT probes |
|---|---|---|---|
| 16   | 0.060 | 0.075 | 0.077 |
| 64   | 0.079 | **0.120** | 0.074 |
| 1025 | 0.250 | **0.500** | 0.298 |

(fraction of oracle power, seed 42; seed 123 agrees to ±0.01)

The covariance of 1025 unknowns is estimated from 600 noisy scenes, so it is
rank-limited. Every LMMSE entry in `paper_benchmark.tex` Table II is
understated, and at M = 1025 by 2×. With a proper covariance LMMSE beats LS
(0.50 > 0.32), as theory requires.

## C. Structured probing: the STORY CHANGES

| M | paper's best (LMMSE) | focus-grid best-probe, **no training** | random best-probe |
|---|---|---|---|
| 16  | 0.060 / 3.18 b/s/Hz | **0.130 / 3.61** | 0.050 / 2.63 |
| 64  | 0.079 / 3.98 | **0.247 / 4.76** | 0.051 / 2.70 |
| 256 | – | 0.305 / 4.73 | 0.052 / 2.52 |
| geometry ceiling (focus on true position) | | **0.912 / 8.25** | |

(fraction of oracle / net bit/s/Hz at L = 2048)

A trivial training-free codebook (focus the surface on a grid of candidate
positions and keep the strongest) beats every scheme in the paper at equal
pilots. Focusing on the true position reaches 91% of oracle. The paper's
claim that acquisition is the binding constraint and LMMSE is the best
deployable scheme was an artifact of random probing.

## Novelty of the corrected story: ALREADY KNOWN

* Near-field RIS beam training beats channel estimation at low pilot
  overhead, and hierarchical codebooks reach ~94% of performance with <1%
  of exhaustive overhead. See arXiv 2310.00294, 2501.02985, 2506.20783 and
  the XL-RIS codebook work.
* The cost of collecting labelled training data is recognised as a problem
  (e.g. active-learning work, arXiv 2402.04896). It is also moot here, because
  the best schemes need no training at all.

## Verdict

Nothing left in the paper is both correct and novel enough for a research
venue. What remains is a correct, reproducible study, with Table II
fixed and codebook baselines added, suitable as a project report or thesis
chapter.

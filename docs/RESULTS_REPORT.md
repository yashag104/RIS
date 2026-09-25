# Results report: what is sound, what is not, and why

**Repository state:** rebased on `62010d5` (full suite regenerated end to end).
**Date:** 2026-09-25.
**Scope:** every figure and every result currently in `results/`, with a verdict on each.

Every number below was read from the regenerated artifacts on this date. Where a
number differs from an earlier document, the earlier document is stale and this
one governs.

---

## How to read the verdicts

| Tag | Meaning |
|---|---|
| **SOUND** | Measured, reproducible, and usable as a claim in a paper today. |
| **SOUND — NEGATIVE** | A correct measurement whose finding is the absence of an effect. Usable, but only for the narrow statement it supports. |
| **VOID** | The measurement executed correctly but the quantity it measures is not the quantity of interest, because a component under test is defective. Cannot support any claim. |
| **INCOMPLETE** | Run not finished. Not citable. |
| **DISCLAIMED** | Conditional calculation under stated assumptions. Usable only with the assumptions attached. |

One rule applied throughout: a result is judged on whether it can bear the weight
of a claim, not on whether it was executed carefully. Several results below were
executed very carefully and are still void.

---

## Part 1 — The sound results

These need no further experiments. They are the publishable core.

### 1.1 Contiguous aperture, validated against the coherent-combining law

**Verdict: SOUND.** This is the strongest result in the project, because it is a
match against a closed-form physical law rather than against another simulation.

Reflected-only received SNR (dB), five-seed mean, `results/link_level_tier2/summary.json`:

| Elements | 64 | 128 | 256 | 512 | 768 | 1024 |
|---|---|---|---|---|---|---|
| Oracle MRC | 2.96 | 8.97 | 15.00 | 21.02 | 24.54 | 27.04 |
| Local MRC | 2.96 | 8.97 | 15.00 | 21.02 | 24.54 | 27.04 |
| N² reference | 3.09 | 9.11 | 15.13 | 21.15 | 24.67 | 27.17 |
| Random phases | −14.61 | −11.82 | −8.60 | −5.52 | −3.61 | −2.39 |

**Why this is good.** Coherent combining of N equal-magnitude terms must give
power proportional to N², i.e. 24.08 dB from 64 to 1024 elements. The measurement
gives 23.99–24.13 dB across seeds. The curve does not bend, does not saturate,
and tracks the law to within a tenth of a decibel over a sixteen-fold increase in
aperture. Tile-average channel power now spreads by 0.06–0.18 dB across the
surface.

**Why it matters beyond itself.** The previous simulator placed the sixteen tiles
on a 6.67 m circle in a 10 m room while the manuscript described a 0.1714 m panel.
That produced a 21.4 dB spread in per-element channel power and a curve that
flattened past 512 elements. The flattening was written up as a physical finding
about tile placement. It was a geometry bug, and this figure is what exposes it.

**Use it as:** a validation figure, not a contribution figure. Its job is to show
the simulator earns trust. Present it as the check that caught the error.

**Figure:** `results/link_level_tier2/seed_*/fig6_array_scaling.pdf`

---

### 1.2 Separability of the single-user problem

**Verdict: SOUND.** Paired differences against local MRC, five seeds,
95% Student-*t* intervals:

| Scheme | Channel power gain (dB) | Paired gap to local MRC (dB) |
|---|---|---|
| Perfect-CSI oracle | −92.328 ± 0.311 | **0.000 ± 0.000** |
| Local noisy-CSI MRC | −92.328 ± 0.311 | 0.000 ± 0.000 |
| Projected-gradient control | −92.349 ± 0.311 | −0.021 ± 0.003 |
| Surrogate control | −92.360 ± 0.310 | −0.032 ± 0.003 |
| Random phases | −106.496 ± 0.800 | −14.167 ± 0.637 |
| No RIS | −106.613 ± 0.811 | −14.284 ± 0.647 |

**Why this is good.** The closed form and the oracle are not merely close — the
paired difference is exactly zero with a zero-width interval, because with perfect
estimates the closed form *is* the oracle. Two iterative optimizers spend many
iterations arriving two to three hundredths of a decibel *below* it. The intervals
on those gaps are three thousandths of a decibel wide, so the ordering is not a
seed accident.

**What it establishes.** For one transmitter, one receiver and known channels, the
optimal phase of each element depends on that element's own coefficient and one
globally shared scalar. Element 900 does not need to know anything about element 12.
Therefore there is nothing for a distributed algorithm to exchange, and nothing for
a learned predictor to discover that a one-line formula does not already give.

**Why it is worth publishing.** It is a boundary result. A substantial literature
proposes iterative, distributed or learned phase design for exactly this setting.
This measurement says: in that setting the problem has one forced answer, so the
interesting question is elsewhere. That is a useful thing for a field to be told,
and it is cheap to verify.

**Use it as:** the paper's pivot. It is what licenses the move to acquisition.

---

### 1.3 The closed form is robust to estimate error

**Verdict: SOUND.** Mean received SNR (dB), five-seed mean, versus normalized
estimate error ε:

| Scheme | ε=0 | 0.01 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|---|
| Oracle (perfect CSI) | 27.67 | 27.67 | 27.67 | 27.67 | 27.67 |
| Local MRC | 27.67 | 27.62 | 27.26 | 26.74 | 25.72 |
| Projected gradient | 27.65 | 27.61 | 27.26 | 26.74 | 25.72 |
| No RIS | 13.39 | 13.39 | 13.39 | 13.39 | 13.39 |

**Why this is good.** At ε=1 the error carries as much power as the signal — half
of what you think you know is wrong — and the closed form surrenders 1.95 dB of the
14.28 dB it had gained. It keeps six-sevenths of its advantage.

**The mechanism, which is the interesting part.** Phase alignment across a thousand
elements is an averaging operation. An estimate error on one element mis-rotates
that element's contribution; that element carries about a thousandth of the total
amplitude. Errors do not compound across the aperture, they wash out. Robustness
here is structural, not incidental.

**Why this is bad news for the learning thesis.** Robustness is only valuable where
fragility exists. Had the closed form been brittle under noisy estimates there
would be an obvious opening for something smarter. There is no fragility to repair.

**A withdrawn claim, and why it was wrong-headed.** The original manuscript reported
that the federated model "degraded most gracefully" — it lost less across this sweep
than the classical solvers. That was arithmetically true and analytically worthless:
it started about two decibels behind and everything converged toward a common floor,
so it had less to lose. Reporting percentage degradation without absolute position is
the same error as praising a team's consistency while ignoring that it finishes tenth
every season. The corrected work reports absolute SNR and fraction-of-oracle-power
for exactly this reason.

**Figure:** `results/link_level_tier2/seed_*/fig5_csi_robustness.pdf` — read the
vertical position first and the slope second.

---

### 1.4 The pilot-budget benchmark: classical estimators

**Verdict: SOUND.** This is the paper's main table. Five seeds, 95% intervals,
net rate is pilot-adjusted with L=2048.

**M = 16 probes** (`results/pilot_limited/summary.json`)

| Scheme | Rx SNR (dB) | Net rate (b/s/Hz) | Frac. oracle power | Paired gap to local linear |
|---|---|---|---|---|
| Perfect-CSI oracle *(cheats)* | 27.672 | 8.388 ± 0.078 | 1.0000 | +5.366 ± 0.073 |
| Full-probe LS + MRC *(1025 probes)* | 25.180 | 3.277 ± 0.059 | 0.3155 | +0.256 ± 0.060 |
| **LMMSE + MRC** | 14.620 | **3.251 ± 0.056** | 0.0581 | +0.230 ± 0.033 |
| Local linear + MRC *(no comms)* | 14.358 | 3.022 ± 0.055 | 0.0559 | 0.000 |
| Best observed probe *(no training)* | 13.767 | 2.649 ± 0.056 | 0.0492 | −0.373 ± 0.074 |
| No RIS | 13.387 | 2.060 ± 0.065 | 0.0454 | −0.961 ± 0.094 |

**M = 64 probes**

| Scheme | Rx SNR (dB) | Net rate (b/s/Hz) | Frac. oracle power |
|---|---|---|---|
| **LMMSE + MRC** | 16.341 | **3.984 ± 0.127** | 0.0774 |
| Local linear + MRC | 15.558 | 3.566 ± 0.132 | 0.0688 |
| Full-probe LS + MRC | 25.180 | 3.277 ± 0.059 | 0.3155 |
| Best observed probe | 13.908 | 2.727 ± 0.039 | 0.0506 |
| No RIS | 13.387 | 2.060 ± 0.065 | 0.0454 |

**Why this is good — three separate findings.**

*First, the acquisition problem is real.* The best deployable scheme reaches under
two-fifths of what perfect knowledge would give. Recovering 1025 complex unknowns
from 16 or 64 measurements is genuinely under-determined, and the gap to the oracle
quantifies how much that costs. This justifies the research question.

*Second, the estimators respond to information properly.* Going from 16 to 64 probes
improves LMMSE by 0.733 b/s/Hz and the per-tile regression by 0.544. Probe-picking
improves by only 0.078, which is exactly what you expect from a scheme that merely
selects a maximum from a longer list. The ordering and the responsiveness both make
physical sense, which is what a benchmark needs to demonstrate before anyone trusts
it.

*Third — and this is the result worth leading with — the accounting reverses the
conclusion.* Exhaustive probing produces by far the best channel knowledge: about
ten decibels of received SNR beyond any competitor, and a fifth of the oracle's
power where the 16-probe methods get a twentieth. Charge it for the half of every
coherence block it consumes and it ties with a 16-probe statistical estimator
(3.277 against 3.251, intervals overlapping). By M=64 it has been *overtaken*
(3.277 against 3.984).

That is a genuine engineering finding and it exists only because pilots were
charged. Report received SNR alone and you would recommend exhaustive probing
without hesitation, and you would be wrong. The crossover between "buy more channel
knowledge" and "keep the block for data" sits between 16 and 1025 probes, and a
64-probe statistical estimator already operates past the sweet spot for exhaustive
probing.

**Figure:** `results/tier01_report/pilot_comparison.pdf` — horizontal bars with
seed-level whiskers. Read in three moves: the oracle bar's distance above everything
(the acquisition problem); the no-RIS bar at the bottom (the floor); then where each
scheme falls between them. Overlapping whiskers mean no claimed difference.

---

### 1.5 The observability certificate

**Verdict: SOUND.** `results/pilot_positive_control/observability_check.json`.
Noiseless algebraic round-trip at M=1025, seed 42, codebook seed 2024:

| Quantity | Value |
|---|---|
| Probe matrix rank | 1025 (full) |
| Condition number | 9.43 × 10³ |
| Relative channel reconstruction error | 3.21 × 10⁻¹³ |
| Fraction of oracle power, float64 inverse | 1.0000000010 |
| **Fraction of oracle power, float32 inverse** | **1.0000000002** |

**Why this is good, and why it is the most methodologically valuable artifact in
the repository.** It answers a question most learning papers never ask: *is the
information the model receives actually sufficient for the task?*

Here the answer is proved rather than assumed. At M=1025 the measurement matrix has
full rank, a plain algebraic inverse of the network's own input recovers the channel
to thirteen decimal places, and the resulting phase configuration attains the oracle
exactly. Critically it still attains it in **32-bit** arithmetic — the precision the
network itself uses — so precision is excluded as an explanation for anything.

**What it lets you conclude.** Two things, and they cut in opposite directions.

It validates the benchmark: the observation model, the feature construction, the
normalization and the phase-application path are all information-preserving. Any
future learning result on this benchmark is measured against a certified-sufficient
input.

And it condemns the learner. Section 2.1 shows a network failing on the same input
from which a matrix inverse extracts perfect performance.

**Use it as:** a methodological contribution. "Certify observability before claiming
a learning result" is a transferable practice, and this is a clean instance of it.

---

### 1.6 Interconnect sensitivity

**Verdict: DISCLAIMED — usable as a sensitivity study, not as a hardware result.**
`results/tier2_report/tables.tex`. Sixteen endpoints, 150,528-byte FP32 model,
20 rounds.

| Topology | Endpoint diameter | ParamServer @128 Gb/s (ms) | RingAllReduce @128 (ms) | RingAllReduce @256 (ms) |
|---|---|---|---|---|
| Mesh | 6 | 5.645520 | 0.716400 | 0.358200 |
| Torus | 4 | 5.645280 | 0.709200 | 0.354600 |
| Folded torus | 4 | 5.645280 | 0.709200 | 0.354600 |
| Tree | 7 | 5.645280 | 1.423800 | 0.711900 |
| **Butterfly** | 6 | 5.645520 | **0.363600** | **0.181800** |
| Hypercube | 4 | 5.645280 | 1.418400 | 0.709200 |
| **Ring** | 8 | 5.645760 | **0.354600** | **0.177300** |

**Why the good part is good.** Two findings, both mechanistically explained.

Under a parameter server all seven topologies cost the same to four decimal places.
The reason is a queue at a single door: every client must reach one designated node,
so the traffic funnels through that node's own port and the network behind it is
irrelevant. You may build whatever road system you like; if everyone passes through
one turnstile, the roads do not matter.

Once traffic is spread across all links, the ranking inverts against intuition. The
plain ring — *longest* worst-case path of all seven — is fastest. The hypercube, with
one of the shortest, is near the bottom. The inference is that network diameter, the
conventional headline metric, measures the wrong thing for this workload: diameter
bounds the longest journey, but bulk-transfer time is set by the *busiest link*. A
ring uses few links and loads every one evenly; a hypercube offers more links and
loads them unevenly. Even beats short.

This directly contradicts the original manuscript, which attributed its speedups to
reduced diameter. The corrected accounting shows the mechanism it claimed was not the
mechanism operating.

**Why it must stay disclaimed.** There is no synthesized circuit behind any of it.
Link width, clock and router pipeline are assumptions; energy is reported as null
because no calibration exists; there is no technology node, no area, no power, and
no cycle-accurate cross-validation. The previous version compounded this by using a
10 Gb/s link rate — an off-chip figure roughly twenty times too slow for on-chip —
which inflated every latency and made the headline speedups look impressive. It also
mislabelled an XOR hypercube as a butterfly. Both are fixed; neither fix turns this
into a hardware contribution.

**Use it as:** a short subsection with the assumptions in the caption, or omit it
from a five-page paper entirely. It is not load-bearing for the benchmark.

---

## Part 2 — The void results

Everything in this part was executed carefully. None of it can support a claim,
because the component under test is defective.

### 2.1 The positive control: the learner fails with complete information

**Verdict: VOID as a learning result. SOUND as a diagnostic.**
`results/pilot_positive_control/summary.json`, M=1025, seeds 42/123/456.

| Scheme | Rx SNR (dB) | Frac. oracle power | Net rate (b/s/Hz) |
|---|---|---|---|
| Perfect-CSI oracle | 27.843 | 1.0000 | 8.419 ± 0.110 |
| Full-probe LS + MRC | 25.435 | 0.3199 | 3.300 ± 0.084 |
| LMMSE + MRC *(linear)* | 23.109 | 0.2534 | 3.166 ± 0.107 |
| Local linear + MRC *(linear)* | 17.865 | 0.0959 | 2.282 ± 0.117 |
| Local models *(neural, unfederated)* | 15.608 | 0.0711 | 1.790 ± 0.071 |
| **FedAvg** *(neural)* | 14.825 | 0.0570 | 1.321 ± 0.025 |
| Centralized *(neural)* | 14.802 | 0.0564 | 1.304 ± 0.026 |
| Best observed probe | 14.414 | 0.0525 | 1.505 ± 0.041 |
| No RIS | 13.706 | 0.0453 | 2.032 ± 0.113 |

**Why this is the decisive result, and why it is bad.**

This experiment removed the difficulty. At M=1025 the system is fully determined —
Section 1.5 proves an algebraic inverse of these exact features attains the oracle
in the network's own precision. Nothing is missing, nothing is ill-conditioned,
nothing is lost to rounding.

The networks still capture under six percent of oracle power. The fair bar is not
the oracle but what a classical method achieves against the *same* pilot noise:
about 32% for least squares, 25% for LMMSE. The networks reach under a fifth of that.

**The single most damning comparison.** `local_linear_mrc` is a ridge regression —
a *linear* map — fitted on identical inputs. It reaches 0.0959 of oracle power.
FedAvg reaches 0.0570. **A linear model beats every neural network by two-thirds
on a task where the optimal map is known to be linear.**

There is a direct road, the driver holds the map, and the driver still does not
arrive. At that point you stop investigating the terrain.

**Worse still, on the headline metric.** The neural schemes' net rate at M=1025
(1.30–1.79) falls *below* switching the surface off (2.032). They pay half the
coherence block in pilots and return less than doing nothing.

**Why the diagnosis is specific rather than vague.** Each tile's network must recover
64 element phases plus the direct path from the probe vector — on the order of 130
real-valued linear functionals of its input. The hidden layer is **128 units wide**.
The bottleneck is narrower than the rank of the map it must represent, so the network
is not failing to optimize; it is structurally unable to express the answer. That
predicts exactly the observed signature: more probes do not help, because the input
was never the constraint.

**What must now be withdrawn.** Every neural claim in the project, including the
federation conclusion. You cannot report "federating does not help" when the learner
fails under every condition including the easiest constructible one. That is a bug
report, not a finding about federated learning.

**What survives.** The classical rows in this table are sound and strengthen
Section 1.4: at full information LMMSE reaches a quarter of oracle power against a
twentieth at M=16, and the per-tile regression nearly doubles. The estimators keep
using information properly.

**The one run that resolves it.** Widen the hidden layer well past the required rank
and repeat this control. If performance climbs toward the classical estimators the
diagnosis is confirmed and the whole suite needs rerunning with adequate capacity.
If it stays flat, the objective's optimization geometry is the problem and that is a
deeper redesign.

---

### 2.2 The main pilot experiment: neural rows

**Verdict: VOID.** From `results/pilot_limited/summary.json`:

| Scheme | Net rate @M=16 | Net rate @M=64 | Change | Frac. oracle @16 |
|---|---|---|---|---|
| LMMSE + MRC | 3.251 | 3.984 | **+0.733** | 0.0581 |
| Local linear + MRC | 3.022 | 3.566 | **+0.544** | 0.0559 |
| Best observed probe | 2.649 | 2.727 | +0.078 | 0.0492 |
| Local models *(neural)* | 2.462 | 2.529 | +0.067 | 0.0480 |
| Centralized *(neural)* | 2.437 | 2.431 | −0.006 | 0.0503 |
| **FedAvg** | 2.347 ± 0.091 | 2.342 ± 0.091 | **−0.005** | 0.0482 |

**What the pattern means.** Quadruple everyone's information. The statistical
estimators improve substantially — they use it. Probe-picking improves slightly, as
expected. The federated network does not move at all, within its interval.

A method that does not improve when given four times more information is not
learning. This is not a weak prior; it is a map roughly independent of its input.
A striker who scores the same on five chances as on twenty is not unlucky.

**The federation-specific observation.** Unfederated local models beat FedAvg at
both budgets (2.462 vs 2.347; 2.529 vs 2.342). Averaging across tiles actively
degraded the models. On a contiguous surface the tiles see nearly identical
statistics, so averaging sixteen models should behave like averaging sixteen noisy
copies of one thing, which normally helps. That it hurts points at the aggregation
step itself — neural networks have internal symmetries, so two networks implementing
similar functions with different internal arrangements average to something between
two valid solutions, which need not be valid. The average of two good routes can run
through a lake. The disclosed optimizer-state reset at each broadcast is the prime
suspect.

**Why it is still void.** Interesting as it is, this observation is about a network
that cannot learn a linear map. It is a property of the broken artefact, not of
federated averaging in general.

**Figure:** `results/tier01_report/validation_curves.pdf`. Loss against round. A
healthy curve drops steeply then flattens — structure found, then exhausted. This one
is nearly flat from the start, drifting down a few percent over a hundred rounds.
That shape is the visual form of the "more data changed nothing" result, and it is
the first figure a sceptical reader will open. **Do not include it in the narrow
paper** without the repaired run beside it; alone it invites the reader to diagnose
the pipeline instead of reading the paper.

---

### 2.3 The architecture diagnostic

**Verdict: VOID.** `results/tier2_report/tables.tex`. M=16, 500 pooled updates,
five seeds.

| Model | Parameters | Rx SNR (dB) | Net rate (b/s/Hz) | Paired gap to MLP |
|---|---|---|---|---|
| CNN + squeeze-excite | 29,730 | 13.997 ± 0.664 | 2.421 ± 0.102 | +0.070 ± 0.063 |
| MLP | 37,632 | 13.719 ± 0.775 | 2.351 ± 0.057 | 0.000 |
| GAT (compact) | 62,850 | 13.994 ± 0.662 | 2.206 ± 0.078 | −0.145 ± 0.031 |
| Transformer | 294,530 | 13.924 ± 0.546 | 2.295 ± 0.081 | −0.056 ± 0.050 |
| *Training-free best probe* | *0* | *13.767* | *2.649 ± 0.056* | *+0.298* |

**What is good about it.** Methodologically it is clean: one shared harness, one
objective, identical data and budget, the same minibatch sequence within a seed,
five seeds with paired intervals. It was built to close the standard escape route
from a failed learning result — "you picked the wrong architecture."

**Why it is nonetheless void.** All four span a band of 0.215 b/s/Hz, narrower than
the 0.298 by which the best of them trails a scheme with no parameters at all. A
ten-fold parameter range changes nothing. Attention — the usual suggestion — finished
*below* the plain network.

But all four share one pipeline. A defect in that shared machinery produces exactly
this signature: identical failure regardless of architecture. Given Section 2.1, that
is now the leading explanation rather than a remote possibility. The experiment
eliminates an alternative explanation; it cannot establish the conclusion, and with
the pipeline known defective it establishes nothing at all.

**Use it as:** evidence, in a future paper, that the defect is upstream of
architecture. Not as an architecture comparison.

**Figure:** `results/tier2_report/pilot_architectures.pdf` — hold it back for the
same reason as the validation curves.

---

### 2.4 The federation communication argument

**Verdict: VOID — the premise does not hold.**
`results/tier01_report/accounting_audit.json`, M=16, seed 42:

| Quantity | Bytes |
|---|---|
| Common receiver feedback (train + validation) | 6,662,400 |
| **Additional upload for centralized training** | **0** |
| Data delivery to tiles for distributed training | 8,294,400 |
| Federated model exchange | 346,816,512 |
| Local models / local MRC model exchange | 0 |

Model-transfer payload runs 26.0–152.8× the size of the entire pooled training set.

**Why the argument fails.** Federation was motivated by avoiding the shipment of raw
measurements to a central point. But in a passive architecture the measurements are
made by the *receiver* and fed back to the controller — that is the only way any of
this functions. The controller therefore already holds everything it needs before
training starts. The extra centralized upload is zero bytes.

So the framework pays a very large communication bill to avoid a cost that was never
going to be charged, and receives worse performance in exchange.

**The privacy fallback also fails.** It needs an adversary. All sixteen tiles are
panels of one surface, owned by one operator, wired to one controller. Keeping data
on-tile protects it from nobody.

---

### 2.5 The corrected-geometry GAT run

**Verdict: INCOMPLETE — not citable.** `results/link_level_corrected_gnn/study_status.json`
records `"status": "running"`, started 2026-09-24T04:59Z, five seeds requested. Only
`seed_42/models.progress.json` exists; no seed result JSON, no aggregate summary.

The status file states its own completion criterion: *"five complete seed result JSON
files and aggregate summary; progress files alone are not evidence."* That criterion
is not met.

This leaves the supplied-CSI table with no neural rows, which is the one Tier 1 item
that remains genuinely open. Given Section 2.1, finishing this run is now lower value
than repairing the learner — it would measure the same defective component on a
different experiment.

---

## Part 3 — Verdict summary

| # | Result | Verdict | Usable in the narrow paper |
|---|---|---|---|
| 1.1 | Aperture validated against N² law | SOUND | Yes — validation figure |
| 1.2 | Separability, paired to zero | SOUND | Yes — the pivot |
| 1.3 | Closed-form robustness to estimate error | SOUND | Yes |
| 1.4 | Pilot-budget classical benchmark | SOUND | Yes — main result |
| 1.5 | Observability certificate | SOUND | Yes — methodology |
| 1.6 | Interconnect sensitivity | DISCLAIMED | Optional, assumptions attached |
| 2.1 | Positive control | VOID / diagnostic | Only as a stated limitation |
| 2.2 | Pilot experiment, neural rows | VOID | No |
| 2.3 | Architecture diagnostic | VOID | No |
| 2.4 | Federation communication argument | VOID | No |
| 2.5 | Corrected-geometry GAT | INCOMPLETE | No |

**Five sound results, one disclaimed, four void, one unfinished.**

The sound five are the expensive half of the work: a validated simulator, a boundary
result, a robustness characterization, a benchmark whose accounting reverses
conclusions, and a reusable observability method. None needs re-running.

The void four all trace to one defect with one specific and testable diagnosis.

---

## Part 4 — What to do, in order

1. **Write the narrow benchmark paper** from the five sound results. Drafted at
   `paper_benchmark.tex`. No new experiments required; the remaining work is related
   work and a compile.

2. **Attempt the capacity repair.** Widen the hidden layer past the rank required
   (well beyond 130 units) and repeat the M=1025 control on three seeds. A few hours.
   This determines whether a second, better paper exists.

3. **If the repair works,** rerun the pilot suite with adequate capacity. Then
   "federation does not help here" becomes defensible, because the learner will have
   been shown to work. That is the stronger paper and it is one repair away.

4. **If the repair fails,** the objective's optimization geometry is the problem.
   Publish the narrow paper and treat the learning question as open.

5. **Do not start multi-user** until one of the above is submitted.

**What to abandon:** the original ambition of a novel federated method that beats the
alternatives. It is not recoverable from this codebase, because the problem it was
built for turned out not to require it.

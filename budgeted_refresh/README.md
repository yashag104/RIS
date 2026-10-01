# Which Elements to Refresh? Bit-Budgeted Reconfiguration of a Large RIS under Channel Aging

**Status (2026-09-29): Gate 1 FAILED — the core claim did not survive the pilot. Do not start writing.**

## Question

The controller has fresh CSI but the control link to the RIS carries only `B`
bits per slot, fewer than the `N*b` a full configuration needs. As the user
moves, which elements should be refreshed each slot, and how much SNR does a
smart choice save over naive refresh?

## Novelty check (done before any code)

| Paper | What it covers | Element-level refresh under a bit budget? |
|---|---|---|
| Enqvist, Demir, Cavdar, Björnson, arXiv 2506.03929 (2025) | Bits for a *full* update; LoS codebook with log N bits; admits codebooks fail in unstructured channels | No. Whole configuration, no aging model |
| Kumar et al., TRACE/DCAR, arXiv 2608.11798 (Aug 2026) | Differential *channel-estimate* tracking from K_D ≪ N probes, Gauss–Markov / random-walk aging | No. All N phases recomputed and applied; no control budget |
| Wu, Chen, Wu, Bai, Wang, arXiv 2507.18727 (2025) | Bit *errors* on the control link, codebook index assignment | No. Whole-surface index; lists time-varying channels as future work |
| arXiv 2603.00407 (2026) | Velocity-aware pilots, progressive element grouping (vehicular) | No. All groups updated each interval; backhaul assumed ideal |
| Koyuncu, Zou, Jafarkhani, arXiv 1808.05410 (2018) | Massive-MIMO analogue: train/feed back antennas one at a time | Closest analogue (MIMO, not RIS control); must be cited |
| LC-RIS transition-aware papers, arXiv 2402.05469, 2504.08352 | Minimise phase *change* for slow LC switching | Different constraint (switching time, not bits) |

Verdict: the specific question was open. Closest prior art to cite: Enqvist
2025, TRACE 2026, Koyuncu 2018.

## Pilot (Gate 1)

`pilot.py` — tiled 32×32 surface at 28 GHz from `src/surface_channel.py`,
straight-line walks (exact near-field LoS drift + Doppler on scatterer paths),
0.5 ms slots, 32 walks × 400 slots, first 150 slots discarded, 1- and 2-bit
phases, speeds 1/3/10 m/s, K = 10 dB and 0 dB. All policies use the same
per-element update rule and differ only in *which* elements they refresh and
*when*. Loss is average-SNR loss against an instantaneous quantised optimum.

**Pre-registered kill criterion:** the best budget-aware policy must beat the
better of full refresh and round-robin by ≥ 1.5 dB at some realistic
(speed, budget).

**Result:** greedy selection's largest win over the better naive policy is
about **1.0–1.3 dB**, and only in rich scattering (K = 0 dB). In the LoS-dominant
case (K = 10 dB) it is ≤ 0.3 dB. **Criterion not met.**

Other findings:

* Sending an index with every element (10 bits for N = 1024) costs more than the
  1–2 bit payload. At low budgets greedy is therefore *worse* than an
  address-free round-robin by up to 5 dB.
* Streaming elements round-robin beats an atomic full refresh by 2–4 dB at
  16–32 bits/slot (K = 10 dB, 3–10 m/s). This is real, but it is an engineering
  observation rather than a result.
* At 10 m/s with K = 0 dB, scatterer paths rotate about 2.9 rad per 0.5 ms slot.
  In that regime a staler, smoothed configuration beats a fresh one. The
  slot length is beyond coherence, so numbers there should not support claims.

Raw numbers: `results/pilot_K10dB.json`, `results/pilot_K0dB.json`.

## Files

* `channel_dynamics.py` — time-varying extension of `src/surface_channel.py`
* `policies.py` — genie (delayed), static, full refresh, round-robin, greedy (indexed), greedy (tile)
* `pilot.py` — Gate-1 experiment

Run: `PYTHONPATH=.:budgeted_refresh .venv/bin/python budgeted_refresh/pilot.py --k-db 10`

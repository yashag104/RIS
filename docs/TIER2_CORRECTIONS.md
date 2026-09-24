# Tier 2 correction audit

Scope: review items 10–16. The research question remains passive receiver feedback under a limited pilot budget. No neural row has been added to the supplied-CSI table. **Tier 1 #8 remains open for the original GAT:** its corrected-geometry training was not run. The pilot MLP plateau evidence and the new finite-budget pilot architecture experiment do not close that item.

| Item | Correction | Evidence and limits |
|---|---|---|
| 10. Link rate / unsupported circuit claims | Default links are 128 bits/cycle at an assumed 1 GHz; the new sensitivity study also evaluates 2 GHz. Removed hardcoded 0.98/0.37 pJ energy constants. Energy/power are null without a sourced calibration. | `src/noc_simulator.py`, `run_noc_study.py`, `results/noc_corrected/results.json`. These are 128/256 Gb/s **assumptions**, not a named-node clock or power result. No RTL, synthesis, area, or Noxim validation exists. The cycle-accurate/discrete-event claim is withdrawn. Naming a process would not manufacture validation. |
| 11. Wrong latency mechanism / missing combination | Price directed-link loads and separate endpoint injection/ejection ports. Report serialization and traversal separately. Evaluate every topology/protocol combination, including Butterfly + RingAllReduce. | ParameterServer's server port dominates these large transfers; diameter does not predict proportional latency reductions. Corrected the XOR-hypercube mislabel and added a staged directed butterfly with 80 auxiliary vertices. No equal-area claim. Folded torus has no invented wire advantage. |
| 12. One seed / unsupported tiny differences | Main wireless studies and the new architecture diagnostic each use five independent seeds, 95% Student-t intervals, and paired differences. | Fixed codebook and synthetic scene model limit interval scope. Architecture intervals are exploratory and marginal. Withdrawn legacy GAT/Transformer and FedAvg/FedProx/SCAFFOLD rankings. Deterministic analytical NoC outputs do not need seed CIs. |
| 13. Incompatible ablation harness / log-only evidence | `prepare_case` in `run_pilot_limited.py` is shared by the headline runner and `run_pilot_ablation.py`; both use the same labels, inputs, phase application, and `score`. | Architecture diagnostic uses 600/200/600 scenes, M=16, Pt=30 dBm, L=2048, five seeds, and 500 pooled Adam updates/model. Per-seed JSON records parameters, validation trajectories, selected steps, feature/codebook hashes and source hashes. `make_tier2_report.py` generates the replacement table and input manifest. No historical log parsing. This centralized, finite-budget diagnostic is not the original GAT convergence experiment. |
| 14. Baseline misattribution | Renamed the primary implementations `ProjectedGradientAscent` and `SISOPhaseSurrogate`. Historical import aliases and JSON keys remain compatible. Removed incorrect MISO/MIMO attribution, beamformer-alternation, compulsory-central-CSI, solver, and KKT claims. | Both are numerical controls for the same separable SISO objective. Local estimated-CSI MRC is the decisive closed-form baseline. SDR and ADMM are not claimed as evaluated controls. |
| 15. Inconsistent axes / power | Active supplied-CSI sweeps use −10 to 30 dBm. Plot primary axes are Pt in dBm; rho remains an internal computation. Both supplied-CSI and pilot operating points are 30 dBm with −90 dBm noise. | `results/link_level_tier2` regenerates all five classical seeds and plots. Unreached BER/rate thresholds stay missing; no extrapolated tens-of-watts power-saving claim. Historical wider sweeps remain archived evidence only. |
| 16. ISCAS scope / length | Reframed as a wireless-systems simulation study; switched to a generic IEEE journal working layout. Circuit novelty, chip power/area, and ISCAS fit are not claimed. | No target submission or acceptance is implied. See venue scope below. The manuscript has not been compiled in this environment, so page-limit compliance remains unverified. |

## Interconnect accounting and assumptions

For each concurrent phase, latency is maximum serialization over all directed links and endpoint ports plus longest-route pipeline traversal. Transfers round up to whole 16-byte flits; remainder chunks are retained in RingAllReduce. Links are full duplex. A two-cycle router and one-cycle link are scenario assumptions. Buffers, credit stalls, arbitration, packet headers, reductions, and physical routing are absent; this is neither a validated bound nor a cycle trace.

The server is on tile 0 in this network study: ParameterServer sends `2*(T-1)*R*model_bytes`. The earlier `2*T*R*model_bytes` endpoint reference assumed an external controller. Both are labelled; they must not be silently equated. A hypothetical colocated controller cluster does not place the RF panel on one die or account for its phase-control wiring.

Recursive doubling exchanges full messages in both directions at each XOR stage. Fixed-budget gossip is not exact global averaging and is not a cheaper equivalent collective. RingAllReduce follows numeric rank order; topology embeddings are not optimized. Logical folded/plain torus graphs coincide because physical layout is not calibrated. Butterfly has extra stage resources and is not an area-normalized competitor.

At 128 Gb/s, 20 rounds with a 150,528-byte MLP take 5.644800 ms of ParameterServer endpoint serialization, independent of topology. Traversal adds 0.000480–0.000960 ms. RingAllReduce estimates depend on routes: ring 0.354600 ms, butterfly 0.363600 ms, torus 0.709200 ms. At 256 Gb/s these values halve under the stated clock-scaled pipeline assumption. These are conditional estimates, not measured training durations. The study also replays actual model sizes and stopping-round counts from all ten retained pilot cases.

The [official Garnet documentation](https://www.gem5.org/documentation/general_docs/ruby/garnet-2/) supports using a configurable 128-bit flit scenario and describes the microarchitectural machinery needed for cycle accuracy. It does not validate this repository's clock, technology node, power, or simplified estimator.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ris-matplotlib .venv/bin/python run_link_level.py --skip-training --output results/link_level_tier2
OPENBLAS_NUM_THREADS=1 .venv/bin/python run_pilot_ablation.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python run_noc_study.py
MPLCONFIGDIR=/tmp/ris-matplotlib .venv/bin/python make_summary_tables.py --link-dir results/link_level_tier2
MPLCONFIGDIR=/tmp/ris-matplotlib .venv/bin/python make_tier2_report.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q test_tier2.py test_noc.py test_tier01.py test_physical_invariants.py test_results_integrity.py test_link_metrics.py test_physics_regression.py
```

The first architecture study's runner was captured unchanged during execution in `results/pilot_architecture/execution_source/`; its execution manifest supplements the shared dependency manifest. The exporter verifies and records that snapshot rather than attributing a later formatting edit to the experiment. No scores are altered by provenance completion.

## Venue scope, checked 2026-09-23

The appropriate category is wireless communications/channel acquisition, not an implemented-circuit paper. The [IEEE ICC Wireless Communications Symposium scope](https://icc2026.ieee-icc.org/sites/icc2026.ieee-icc.org/files/ICC%202026%20CFP_Wireless%20Communications%20Symposium.pdf) is a useful scope reference for a future conference edition; the linked 2026 event is historical, not an open deadline recommendation. A concise new contribution could be considered for [IEEE Wireless Communications Letters](https://www.comsoc.org/publications/journals/ieee-wcl/policies-guidelines), whose current instructions cap letters at five pages. A fuller investigation is closer in scope to [IEEE Transactions on Wireless Communications](https://www.comsoc.org/publications/journals/ieee-twc); its [current submission guidelines](https://www.comsoc.org/publications/journals/ieee-twc/policies-guidelines) cap initial transactions manuscripts at 13 double-column pages. These are scope options, not predictions of acceptance.

The present evidence still shows no useful FL advantage over local/statistical controls. Correcting the code and reporting does not supply a novel beneficial federated method. A submission must make a defensible contribution from the actual findings, establish its novelty in the literature, and then meet the chosen venue's format. This work does not invent a hardware implementation or disguise negative findings to broaden venue eligibility.

## Completion and verification (2026-09-24)

All five supplied-CSI seeds, twenty architecture/seed combinations, and 672 analytical interconnect cases completed. The supplied-CSI operating-point gains are unchanged from the corrected Tier 0/1 audit; the changed power window affects the sweeps, not the channels. All six classical controls in the refactored pilot harness reproduce their original seed-42 scores within 1e-10 relative tolerance.

The main regression suite passed 74 tests with one optional solver test skipped. The final focused Tier 2/NoC suite passed 12 tests. Targeted Ruff checks and `git diff --check` pass. Export input hashes and NoC source/input hashes were checked. New plots were visually inspected. The legacy live-Config watermark incorrectly labelled the full classical audit as a reduced training run; corrected plots use per-artifact `figure_manifest.json` records with actual scene/seed/training status and source, renderer, and image hashes instead. Unreached-target panels have no misleading numeric axis. The current manuscript has not been compiled because no LaTeX compiler is installed.

The new compact GAT has a paired net-rate gap to MLP of −0.156 ± 0.054 bit/s/Hz at 500 pooled updates. CNN and Transformer intervals versus MLP include zero. These are finite-budget pilot results, not supplied-CSI GAT evidence, convergence proofs, or a demonstrated FL advantage.

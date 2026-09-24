# Tier 2 corrected results

Architecture diagnostic: five seeds, 500 pooled Adam updates each; M=16, Pt=30 dBm, L=2048.
600/200/600 train/validation/test scenes; shared pilot acquisition and scoring harness.
Validation-selected checkpoints. Budget-limited, unequal FLOPs/parameter counts; no convergence or supplied-CSI GAT claim.

| Pilot model | Parameters | Received SNR (dB), 95% CI | Net SE (bit/s/Hz), 95% CI | Paired net-SE gap vs MLP |
|---|---:|---:|---:|---:|
| MLP | 37632 | 13.719 ± 0.775 | 2.351 ± 0.057 | 0.000 ± 0.000 |
| GAT (compact) | 62850 | 13.994 ± 0.662 | 2.206 ± 0.078 | -0.145 ± 0.031 |
| CNN + SE | 29730 | 13.997 ± 0.664 | 2.421 ± 0.102 | 0.070 ± 0.063 |
| Transformer | 294530 | 13.924 ± 0.546 | 2.295 ± 0.081 | -0.056 ± 0.050 |

Intervals are marginal/exploratory, not simultaneous multiple-comparison intervals. No universal architecture ranking is inferred.

Analytical interconnect sensitivity: 16 tiles, 150,528-byte FP32 model, 20 rounds.
128-bit links and endpoint ports at assumed 1/2 GHz (128/256 Gb/s); server at tile 0.
No cycle accuracy, Noxim validation, physical layout, energy, area, or technology-node result.

| Logical topology | Endpoint diameter | PS at 128 Gb/s (ms) | RingAR at 128 Gb/s (ms) | RingAR at 256 Gb/s (ms) |
|---|---:|---:|---:|---:|
| Mesh | 6 | 5.645520 | 0.716400 | 0.358200 |
| Torus | 4 | 5.645280 | 0.709200 | 0.354600 |
| FoldedTorus | 4 | 5.645280 | 0.709200 | 0.354600 |
| Tree | 7 | 5.645280 | 1.423800 | 0.711900 |
| Butterfly | 6 | 5.645520 | 0.363600 | 0.181800 |
| Hypercube | 4 | 5.645280 | 1.418400 | 0.709200 |
| Ring | 8 | 5.645760 | 0.354600 | 0.177300 |

PS serialization alone is 5.644800 ms for every topology at 128 Gb/s; traversal adds 0.000480–0.000960 ms.
A smaller diameter does not remove the endpoint payload. Ring placement/routing changes physical link contention.
Torus and folded torus coincide without a physical wire-layout model. Butterfly has 80 auxiliary stage vertices; it is not the former mislabeled 16-node hypercube.
The complete JSON also replays every retained pilot run's measured model size and FL round count, and includes recursive-doubling and fixed-budget gossip. Gossip is not exact global averaging.

Tier 1 #8 remains open for the original supplied-CSI GAT. The retained pilot MLP runs have operational validation plateaus only.

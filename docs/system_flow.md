# System Flow

What this project builds, and how every piece connects.

---

## 1. The problem

A base station wants to serve a user whose direct path is blocked. A
Reconfigurable Intelligent Surface on the wall reflects the signal around the
obstruction. The surface only helps if each of its elements applies the right
phase shift, so the phases must be chosen from channel state information.

The surface here is large: **1024 elements**, arranged as **16 tiles of 64
pixels** each. Each tile carries its own small processor, and the tiles are
wired together by an on-chip network.

```mermaid
flowchart LR
    BS["Base station<br/>28 GHz"]
    RIS["RIS: 16 tiles x 64 pixels<br/>1024 elements total"]
    UE["User"]
    BLK["Obstruction<br/>30 dB blockage"]

    BS -. "direct path, blocked" .-> BLK
    BLK -. "weak residual" .-> UE
    BS -- "h_bs_ris" --> RIS
    RIS -- "h_ris_user, phase-shifted" --> UE

    style BLK fill:#f8d7da,stroke:#c33
    style RIS fill:#d1e7dd,stroke:#2a7
```

The received signal is the sum of the blocked direct path and the reflection:

```
h_total = h_direct + SUM_over_elements( h_cascade * exp(j * phase) )
h_cascade = h_ris_user * h_bs_ris
```

The best possible phases align every reflected contribution with the direct
path. That maximum-ratio combining solution is the **genie-aided optimum**, and
it is the upper bound nothing may exceed.

---

## 2. The central question

Conventional designs ship all channel state information to one central
optimizer. That costs bandwidth, leaks the user's channel, and scales badly.

**This project asks whether the tiles can instead learn the phase-prediction
model cooperatively, each keeping its own data, exchanging only model weights
over the on-chip network.**

```mermaid
flowchart TB
    subgraph CENTRAL["Centralized: what we compare against"]
        direction LR
        C1["Tile 1 raw CSI"] --> CS["Central optimizer"]
        C2["Tile 2 raw CSI"] --> CS
        C3["Tile N raw CSI"] --> CS
        CS --> CO["Phases"]
    end

    subgraph FED["Federated: what we propose"]
        direction LR
        F1["Tile 1<br/>trains locally"] <--> FS["Aggregator<br/>averages weights"]
        F2["Tile 2<br/>trains locally"] <--> FS
        F3["Tile N<br/>trains locally"] <--> FS
    end

    CENTRAL --- NOTE1["Raw CSI leaves the tile<br/>Best accuracy, worst privacy"]
    FED --- NOTE2["Only weights leave the tile<br/>Question: what does this cost?"]

    style CENTRAL fill:#fff3cd,stroke:#b8860b
    style FED fill:#d1e7dd,stroke:#2a7
```

Answering it honestly requires measuring the accuracy given up, the on-chip
traffic and energy spent, and the robustness gained or lost. That is what the
twenty experiments do.

---

## 3. End-to-end pipeline

```mermaid
flowchart TB
    CFG["config.py<br/>geometry, channel, FL, NoC settings"]

    subgraph GEN["Channel generation - src/channel_model.py"]
        RIC["RicianChannel<br/>K-factor, spatial correlation"]
        GPP["ThreeGPPUMiChannel<br/>TR 38.901 UMi"]
        SCENE["generate_multi_tile_channels<br/>ONE shared scene, all tiles"]
        RIC --> SCENE
        GPP --> SCENE
    end

    subgraph DATA["Dataset - src/dataset_utils.py"]
        TRAIN["create_non_iid_datasets<br/>one dataset per tile"]
        TEST["create_test_dataset<br/>held-out evaluation set"]
        LBL["Labels = MRC-optimal phases"]
    end

    subgraph FL["Federated training"]
        CLI["RISClient x 16<br/>src/client.py"]
        SRV["FederatedServer<br/>src/server.py"]
        NOC["NoCSimulator<br/>latency, energy, congestion"]
    end

    EVAL["Evaluation<br/>SNR, phase error, accuracy"]
    BASE["Baselines<br/>baselines/*.py"]
    EXP["20 experiments<br/>experiments/*.py"]
    OUT["Results JSON + IEEE figures"]

    CFG --> GEN --> DATA --> FL
    LBL --> FL
    FL --> EVAL
    BASE --> EVAL
    EVAL --> EXP --> OUT

    style CFG fill:#e7e7ff,stroke:#66c
    style OUT fill:#d1e7dd,stroke:#2a7
```

**Why the scene is shared.** Every tile must be illuminated by the *same* drawn
user and the same direct path, otherwise tile 1's sample and tile 2's sample
describe different worlds and their reflections cannot be summed into one
1024-element surface.

---

## 4. One federated round

```mermaid
sequenceDiagram
    participant S as Server
    participant N as NoC
    participant T as Tiles 1..16

    S->>N: broadcast global weights
    N->>T: deliver, cost priced by topology and protocol
    loop LOCAL_EPOCHS = 3
        T->>T: train on local channels
    end
    T->>N: upload updated weights
    N->>S: deliver, cost priced again
    S->>S: aggregate: FedAvg, FedProx or SCAFFOLD
    S->>S: evaluate global model on held-out test set
    Note over S: repeat for FL_ROUNDS = 20
```

Model choices are `GNN` by default, with `MLP`, `CNN_Attention` and
`Transformer` available. The training objective is sum rate by default, with
mean-squared-error on phases and raw SNR as alternatives.

The network layer is not a formula. `NoCSimulator` routes every transfer over
the chosen topology and prices the round by its most loaded link.

```mermaid
flowchart LR
    subgraph T["Topologies"]
        M[Mesh] ~~~ TO[Torus] ~~~ FT[FoldedTorus]
        TR[Tree] ~~~ BF[Butterfly] ~~~ RG[Ring]
    end
    subgraph P["Protocols"]
        PS[ParameterServer] ~~~ AR[AllReduce]
        RA[RingAllReduce] ~~~ GO[Gossip]
    end
    T --> COST["Per round:<br/>latency, energy,<br/>utilization, congestion"]
    P --> COST
```

---

## 5. The measurement path

Every claim reduces to comparing SNR under different phase choices. **All arms
must be scored on the same channel realisations**, or the comparison is noise.

```mermaid
flowchart TB
    CH["One test channel sample<br/>h_direct, h_cascade"]

    CH --> P1["No RIS<br/>direct path only"]
    CH --> P2["Random phases"]
    CH --> P3["Classical optimizers<br/>AO, SCA, ADMM, SDR"]
    CH --> P4["Learned model<br/>federated or centralized"]
    CH --> P5["Genie MRC optimum"]

    P1 --> SNR["SNR = P_tx * abs h_total squared / N_0"]
    P2 --> SNR
    P3 --> SNR
    P4 --> SNR
    P5 --> SNR

    SNR --> GUARD{"Does any arm<br/>exceed the genie?"}
    GUARD -- yes --> FAIL["FAIL LOUDLY<br/>arms are on different data"]
    GUARD -- no --> OK["Report"]

    style P5 fill:#fff3cd,stroke:#b8860b
    style FAIL fill:#f8d7da,stroke:#c33
    style OK fill:#d1e7dd,stroke:#2a7
```

The genie check is enforced in code, not by inspection. It exists because the
federated arm once scored *above* the oracle: the two were being evaluated on
independently drawn channels, and draw-to-draw spread is several dB, larger than
the effects being measured.

---

## 6. Experiment matrix

Twenty experiments, driven by `run_all_experiments.py`, grouped by what they
vary.

```mermaid
flowchart LR
    subgraph LEARN["Learning"]
        E1["1 Local epochs"]
        E3["3 Model compression"]
        E5["5 Non-IID heterogeneity"]
        E11["11 FL algorithms"]
        E12["12 Model architectures"]
    end
    subgraph PHYS["Physics and robustness"]
        E2["2 RIS quantization levels"]
        E4["4 User mobility"]
        E13["13 CSI robustness"]
        E19["19 Dataset comparison"]
        E20["20 Phase quantization"]
    end
    subgraph HW["Hardware and network"]
        E6["6 Pilot overhead"]
        E7["7 NoC traffic vs power"]
        E14["14 NoC topology"]
        E15["15 Protocols"]
        E17["17 Tile-pixel ratio"]
        E18["18 Duty cycling"]
    end
    subgraph COMP["Comparisons"]
        E8["8 FL vs centralized"]
        E9["9 Baseline comparison"]
        E10["10 Multi-user MIMO"]
        E16["16 Optimization techniques"]
    end
```

---

## 7. Integrity controls

Results that look plausible and mean nothing are the main risk in this codebase,
so several checks are wired into the pipeline itself.

```mermaid
flowchart TB
    RUN["Experiment run"]
    SEED["Per-experiment seed<br/>derived from Config.SEED"]
    PROV["Provenance block on every JSON<br/>rounds, samples, seeds, key settings"]
    RED["is_reduced_run flag<br/>marks smoke runs"]
    STAMP["Figure stamp<br/>date, scale, seed on the image"]
    ORACLE["Genie upper-bound assertion"]
    TESTS["test_results_integrity.py<br/>regression tests"]

    SEED --> RUN --> PROV --> RED
    RUN --> STAMP
    RUN --> ORACLE
    TESTS -.guards.-> RUN

    style ORACLE fill:#fff3cd,stroke:#b8860b
    style TESTS fill:#d1e7dd,stroke:#2a7
```

| Control | Failure it prevents |
|---|---|
| Per-experiment seeding | Irreproducible runs |
| Provenance block | A 5-round smoke run passing as a full run |
| Figure stamp | A figure replotted from stale data looking fresh |
| Genie assertion | Arms compared on different channel draws |
| Measured-not-assumed rule | A closed form published as a measurement |
| Regression tests | Any of the above returning silently |

The last two exist because both happened. A fairness index was assigned as
`0.5 + 0.4 * alpha` and plotted as data, and a mobility "adaptation time" was
assigned as `5 + 2 * speed`. Neither ever touched the simulation.

---

## 8. Where things live

| Path | Role |
|---|---|
| `config.py` | Single source of truth for every parameter |
| `src/channel_model.py` | Rician and 3GPP channels, quantization, CSI error |
| `src/dataset_utils.py` | Per-tile datasets, held-out test set, partitioning |
| `src/client.py` | Tile-local training and SNR evaluation |
| `src/server.py` | FedAvg, FedProx, SCAFFOLD aggregation |
| `src/noc_simulator.py` | Topologies, protocols, latency and energy |
| `models/ris_net.py` | GNN, MLP, CNN+Attention, Transformer |
| `baselines/` | AO, SCA, ADMM, SDR, DRL, random search, centralized |
| `experiments/` | The twenty experiments |
| `utils/plotting*.py` | IEEE-style figures |
| `run_all_experiments.py` | Orchestration, seeding, progress, campaigns |

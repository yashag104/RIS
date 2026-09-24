# Tier 0 / Tier 1 corrected results

Supplied-CSI audit: 5 independent seeds.

| Scheme | Channel gain (dB), 95% CI |
|---|---:|
| No RIS | -106.61 ± 0.81 |
| Random phases | -106.50 ± 0.80 |
| Local noisy-CSI MRC | -92.33 ± 0.31 |
| Projected-gradient control | -92.35 ± 0.31 |
| Surrogate control | -92.36 ± 0.31 |
| Perfect-CSI bound | -92.33 ± 0.31 |

Tile-average power spread by seed (dB): [0.1783722061605033, 0.11583869702081984, 0.11811632995258492, 0.059922008400519644, 0.08276039440343652].
Reflected-only oracle gain from 64 to 1024 elements (dB): [23.98625829876947, 24.07584482125459, 24.12729479223666, 24.07876681870187, 24.101914621844365].


## M16_Pt30

| Scheme | SNR (dB) | Net rate | Paired gap to local linear |
|---|---:|---:|---:|
| No RIS | 13.387 ± 0.811 | 2.060 ± 0.065 | -0.961 ± 0.094 |
| Best observed probe | 13.767 ± 0.770 | 2.649 ± 0.056 | -0.373 ± 0.074 |
| LMMSE + MRC | 14.620 ± 0.728 | 3.251 ± 0.056 | 0.230 ± 0.033 |
| Local linear estimate + MRC | 14.358 ± 0.764 | 3.022 ± 0.055 | 0.000 ± 0.000 |
| Full-probe LS + MRC | 25.180 ± 0.473 | 3.277 ± 0.059 | 0.256 ± 0.060 |
| Perfect-CSI bound (oracle) | 27.672 ± 0.311 | 8.388 ± 0.078 | 5.366 ± 0.073 |
| One-round FedAvg | 13.506 ± 0.763 | 2.219 ± 0.075 | -0.803 ± 0.094 |
| Five-round FedAvg | 13.584 ± 0.799 | 2.300 ± 0.078 | -0.722 ± 0.078 |
| FedAvg (validation selected) | 13.692 ± 0.780 | 2.351 ± 0.099 | -0.671 ± 0.120 |
| Central (client-step budget) | 13.762 ± 0.810 | 2.371 ± 0.081 | -0.651 ± 0.094 |
| Central (total-step budget) | 13.816 ± 0.829 | 2.403 ± 0.028 | -0.618 ± 0.043 |
| Local models | 13.657 ± 0.793 | 2.462 ± 0.053 | -0.560 ± 0.081 |

## M64_Pt30

| Scheme | SNR (dB) | Net rate | Paired gap to local linear |
|---|---:|---:|---:|
| No RIS | 13.387 ± 0.811 | 2.060 ± 0.065 | -1.506 ± 0.139 |
| Best observed probe | 13.908 ± 0.725 | 2.727 ± 0.039 | -0.839 ± 0.109 |
| LMMSE + MRC | 16.341 ± 0.739 | 3.984 ± 0.127 | 0.418 ± 0.053 |
| Local linear estimate + MRC | 15.558 ± 0.727 | 3.566 ± 0.132 | 0.000 ± 0.000 |
| Full-probe LS + MRC | 25.180 ± 0.473 | 3.277 ± 0.059 | -0.289 ± 0.103 |
| Perfect-CSI bound (oracle) | 27.672 ± 0.311 | 8.388 ± 0.078 | 4.821 ± 0.100 |
| One-round FedAvg | 13.539 ± 0.711 | 2.188 ± 0.056 | -1.378 ± 0.143 |
| Five-round FedAvg | 13.618 ± 0.719 | 2.260 ± 0.053 | -1.306 ± 0.143 |
| FedAvg (validation selected) | 13.792 ± 0.854 | 2.323 ± 0.057 | -1.243 ± 0.148 |
| Central (client-step budget) | 13.829 ± 0.699 | 2.360 ± 0.059 | -1.206 ± 0.129 |
| Central (total-step budget) | 14.031 ± 0.816 | 2.412 ± 0.106 | -1.154 ± 0.131 |
| Local models | 13.754 ± 0.769 | 2.527 ± 0.081 | -1.039 ± 0.113 |

Training stopping and payload accounting:

```json
[
  {
    "seed": 42,
    "M": 16,
    "pt_dbm": 30.0,
    "rounds": 72,
    "best_round": 57,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 346816512,
    "pooled_dataset_payload_reference_bytes": 4996800,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 123,
    "M": 16,
    "pt_dbm": 30.0,
    "rounds": 55,
    "best_round": 41,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 264929280,
    "pooled_dataset_payload_reference_bytes": 4996800,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 456,
    "M": 16,
    "pt_dbm": 30.0,
    "rounds": 45,
    "best_round": 30,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 216760320,
    "pooled_dataset_payload_reference_bytes": 4996800,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 789,
    "M": 16,
    "pt_dbm": 30.0,
    "rounds": 32,
    "best_round": 18,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 154140672,
    "pooled_dataset_payload_reference_bytes": 4996800,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 1024,
    "M": 16,
    "pt_dbm": 30.0,
    "rounds": 27,
    "best_round": 12,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 130056192,
    "pooled_dataset_payload_reference_bytes": 4996800,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 42,
    "M": 64,
    "pt_dbm": 30.0,
    "rounds": 90,
    "best_round": 75,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 575078400,
    "pooled_dataset_payload_reference_bytes": 5227200,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 123,
    "M": 64,
    "pt_dbm": 30.0,
    "rounds": 123,
    "best_round": 109,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 785940480,
    "pooled_dataset_payload_reference_bytes": 5227200,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 456,
    "M": 64,
    "pt_dbm": 30.0,
    "rounds": 122,
    "best_round": 107,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 779550720,
    "pooled_dataset_payload_reference_bytes": 5227200,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 789,
    "M": 64,
    "pt_dbm": 30.0,
    "rounds": 84,
    "best_round": 69,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 536739840,
    "pooled_dataset_payload_reference_bytes": 5227200,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 1024,
    "M": 64,
    "pt_dbm": 30.0,
    "rounds": 44,
    "best_round": 29,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 281149440,
    "pooled_dataset_payload_reference_bytes": 5227200,
    "additional_centralized_training_upload_bytes": 0
  }
]
```

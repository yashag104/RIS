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
| Five-round FedAvg | 13.588 ± 0.782 | 2.296 ± 0.075 | -0.726 ± 0.080 |
| FedAvg (validation selected) | 13.673 ± 0.798 | 2.347 ± 0.091 | -0.674 ± 0.107 |
| Central (client-step budget) | 13.715 ± 0.801 | 2.362 ± 0.068 | -0.659 ± 0.088 |
| Central (total-step budget) | 13.816 ± 0.851 | 2.437 ± 0.050 | -0.585 ± 0.053 |
| Local models | 13.659 ± 0.765 | 2.462 ± 0.055 | -0.559 ± 0.087 |

## M64_Pt30

| Scheme | SNR (dB) | Net rate | Paired gap to local linear |
|---|---:|---:|---:|
| No RIS | 13.387 ± 0.811 | 2.060 ± 0.065 | -1.506 ± 0.139 |
| Best observed probe | 13.908 ± 0.725 | 2.727 ± 0.039 | -0.839 ± 0.109 |
| LMMSE + MRC | 16.341 ± 0.739 | 3.984 ± 0.127 | 0.418 ± 0.053 |
| Local linear estimate + MRC | 15.558 ± 0.727 | 3.566 ± 0.132 | 0.000 ± 0.000 |
| Full-probe LS + MRC | 25.180 ± 0.473 | 3.277 ± 0.059 | -0.289 ± 0.103 |
| Perfect-CSI bound (oracle) | 27.672 ± 0.311 | 8.388 ± 0.078 | 4.821 ± 0.100 |
| One-round FedAvg | 13.539 ± 0.710 | 2.188 ± 0.056 | -1.378 ± 0.143 |
| Five-round FedAvg | 13.616 ± 0.718 | 2.260 ± 0.048 | -1.306 ± 0.137 |
| FedAvg (validation selected) | 13.780 ± 0.750 | 2.343 ± 0.092 | -1.223 ± 0.152 |
| Central (client-step budget) | 13.823 ± 0.793 | 2.346 ± 0.093 | -1.220 ± 0.114 |
| Central (total-step budget) | 13.961 ± 0.713 | 2.427 ± 0.090 | -1.140 ± 0.142 |
| Local models | 13.773 ± 0.738 | 2.529 ± 0.109 | -1.038 ± 0.126 |

Training stopping and payload accounting:

```json
[
  {
    "seed": 42,
    "M": 16,
    "pt_dbm": 30.0,
    "rounds": 63,
    "best_round": 48,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 303464448,
    "pooled_dataset_payload_reference_bytes": 4996800,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 123,
    "M": 16,
    "pt_dbm": 30.0,
    "rounds": 65,
    "best_round": 50,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 313098240,
    "pooled_dataset_payload_reference_bytes": 4996800,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 456,
    "M": 16,
    "pt_dbm": 30.0,
    "rounds": 44,
    "best_round": 29,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 211943424,
    "pooled_dataset_payload_reference_bytes": 4996800,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 789,
    "M": 16,
    "pt_dbm": 30.0,
    "rounds": 100,
    "best_round": 92,
    "stopping_reason": "budget_exhausted",
    "fl_bytes": 481689600,
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
    "rounds": 81,
    "best_round": 66,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 517570560,
    "pooled_dataset_payload_reference_bytes": 5227200,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 123,
    "M": 64,
    "pt_dbm": 30.0,
    "rounds": 125,
    "best_round": 125,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 798720000,
    "pooled_dataset_payload_reference_bytes": 5227200,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 456,
    "M": 64,
    "pt_dbm": 30.0,
    "rounds": 73,
    "best_round": 58,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 466452480,
    "pooled_dataset_payload_reference_bytes": 5227200,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 789,
    "M": 64,
    "pt_dbm": 30.0,
    "rounds": 116,
    "best_round": 101,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 741212160,
    "pooled_dataset_payload_reference_bytes": 5227200,
    "additional_centralized_training_upload_bytes": 0
  },
  {
    "seed": 1024,
    "M": 64,
    "pt_dbm": 30.0,
    "rounds": 105,
    "best_round": 90,
    "stopping_reason": "validation_plateau",
    "fl_bytes": 670924800,
    "pooled_dataset_payload_reference_bytes": 5227200,
    "additional_centralized_training_upload_bytes": 0
  }
]
```

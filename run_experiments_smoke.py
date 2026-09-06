#!/usr/bin/env python
"""Run the experiment suite one at a time at reduced scale and report which work.

Purpose is to verify every experiment EXECUTES correctly end to end after the
channel/model/objective corrections -- not to produce publication-scale numbers.
The default config (2000 samples x 20 rounds x 16 tiles) needs days on CPU;
these settings finish in minutes while exercising exactly the same code paths.

Usage:
    python run_experiments_smoke.py                 # experiments 1-20
    python run_experiments_smoke.py 9 10            # only the listed ones
    python run_experiments_smoke.py 11-20           # an inclusive range
    python run_experiments_smoke.py 11-20 --skip 18 # range minus specific ones

Experiment 18 (duty cycling) is skipped by default: the duty-cycling mechanism
is deliberately left untouched, and its energy/SNR trade-off is already
documented from direct measurement in config.py. Pass it explicitly to run it.
"""
import json
import sys
import time
import traceback

import numpy as np

from config import Config

# ---- Reduced scale. Same code paths, far less compute. ----
Config.NUM_TILES = 4
Config.TILE_GRID_ROWS, Config.TILE_GRID_COLS = 2, 2
Config.TRAIN_SAMPLES = 150
Config.TEST_SAMPLES = 150
Config.FL_ROUNDS = 4
Config.LOCAL_EPOCHS = 1
Config.BATCH_SIZE = 32

from experiments.logging_utils import (  # noqa: E402
    configure_experiment_logging,
    resolve_log_level,
)
from experiments.suite import AdvancedExperiments  # noqa: E402

OUT_JSON_TEMPLATE = "results/experiments_smoke_{tag}.json"


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def main(argv):
    configure_experiment_logging(
        f"{Config.RESULTS_DIR}/advanced_experiments", level=resolve_log_level(Config)
    )
    exp = AdvancedExperiments(Config)

    runs = [
        (1, 'local_epochs', exp.experiment_1_local_epochs_variation),
        (2, 'quantization', exp.experiment_2_quantization_levels),
        (3, 'compression', exp.experiment_3_model_compression),
        (4, 'mobility', exp.experiment_4_user_mobility),
        (5, 'non_iid', exp.experiment_5_non_iid_heterogeneity),
        (6, 'pilots', exp.experiment_6_pilot_overhead),
        (7, 'noc', exp.experiment_7_noc_traffic_vs_power),
        (8, 'fed_vs_central', exp.experiment_8_federated_vs_centralized),
        (9, 'baselines', exp.experiment_9_baseline_comparison),
        (10, 'multiuser', exp.experiment_10_multiuser_comparison),
        (11, 'fl_algorithms', exp.experiment_11_fl_algorithms),
        (12, 'architectures', exp.experiment_12_architectures),
        (13, 'csi_robustness', exp.experiment_13_csi_robustness),
        (14, 'topology', exp.experiment_14_topology_comparison),
        (15, 'protocol', exp.experiment_15_protocol_comparison),
        (16, 'optimization', exp.experiment_16_optimization_techniques),
        (17, 'golden_ratio', exp.experiment_17_tile_pixel_golden_ratio),
        (18, 'duty_cycling', exp.experiment_18_duty_cycling),
        (19, 'datasets', exp.experiment_19_dataset_comparison),
        (20, 'phase_quantization', exp.experiment_20_phase_quantization),
    ]

    # Experiment 18 is skipped unless asked for by number; the duty-cycling
    # mechanism is intentionally left as-is.
    DEFAULT_SKIP = {18}

    explicit, skip = [], set()
    it = iter(argv)
    for arg in it:
        if arg == '--skip':
            skip |= {int(x) for x in next(it, '').split(',') if x}
        elif '-' in arg and not arg.startswith('-'):
            lo, hi = arg.split('-')
            explicit += list(range(int(lo), int(hi) + 1))
        else:
            explicit.append(int(arg))

    if explicit:
        wanted = set(explicit) - skip
    else:
        wanted = {n for n, _, _ in runs} - DEFAULT_SKIP - skip

    summary, results = [], {}
    for num, name, fn in runs:
        if num not in wanted:
            continue
        print(f"\n{'#' * 72}\n### EXPERIMENT {num}: {name}\n{'#' * 72}", flush=True)
        t0 = time.time()
        try:
            res = fn()
            dt = time.time() - t0
            results[name] = res
            summary.append((num, name, 'OK', dt, ''))
            print(f"### EXP {num} ({name}) OK in {dt:.0f}s", flush=True)
        except Exception as exc:
            dt = time.time() - t0
            summary.append((num, name, 'FAILED', dt, f"{type(exc).__name__}: {exc}"))
            print(f"### EXP {num} ({name}) FAILED after {dt:.0f}s: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()

    print(f"\n{'=' * 72}\nSUMMARY", flush=True)
    for num, name, status, dt, err in summary:
        print(f"  exp {num:>2}  {name:<16} {status:<7} {dt:>6.0f}s  {err}", flush=True)
    ok = sum(1 for s in summary if s[2] == 'OK')
    print(f"\n  {ok}/{len(summary)} experiments completed", flush=True)

    # Tag the output by which experiments ran, so a 11-20 run does not clobber
    # the 1-10 results.
    nums = sorted(n for n, *_ in summary)
    tag = f"{nums[0]}-{nums[-1]}" if nums else "none"
    out_json = OUT_JSON_TEMPLATE.format(tag=tag)

    with open(out_json, 'w') as fh:
        json.dump({'summary': [
            {'num': n, 'name': nm, 'status': st, 'seconds': dt, 'error': e}
            for n, nm, st, dt, e in summary
        ], 'results': _jsonable(results)}, fh, indent=2)
    print(f"  wrote {out_json}", flush=True)
    return 0 if ok == len(summary) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

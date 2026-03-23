import argparse
import os
import sys
import time
import json
import traceback

from config import Config
from experiments import AdvancedExperiments

EXPERIMENTS = {
    1:  ('Local Epochs Variation',       'experiment_1_local_epochs_variation'),
    2:  ('RIS Quantization Levels',      'experiment_2_quantization_levels'),
    3:  ('Model Compression',            'experiment_3_model_compression'),
    4:  ('User Mobility',                'experiment_4_user_mobility'),
    5:  ('Non-IID Heterogeneity',        'experiment_5_non_iid_heterogeneity'),
    6:  ('Pilot Overhead',               'experiment_6_pilot_overhead'),
    7:  ('NoC Traffic vs Power',         'experiment_7_noc_traffic_vs_power'),
    8:  ('FL vs Centralized',            'experiment_8_federated_vs_centralized'),
    9:  ('Baseline Comparison',          'experiment_9_baseline_comparison'),
    10: ('Multi-User MIMO',              'experiment_10_multiuser_comparison'),
    11: ('FL Algorithms',                'experiment_11_fl_algorithms'),
    12: ('Model Architectures',          'experiment_12_architectures'),
    13: ('CSI Robustness',               'experiment_13_csi_robustness'),
    14: ('NoC Topology Comparison',      'experiment_14_topology_comparison'),
    15: ('Communication Protocols',      'experiment_15_protocol_comparison'),
    16: ('Optimization Techniques',      'experiment_16_optimization_techniques'),
    17: ('Tile-Pixel Golden Ratio',      'experiment_17_tile_pixel_golden_ratio'),
    18: ('Dynamic Duty Cycling',         'experiment_18_duty_cycling'),
    19: ('Dataset Comparison',           'experiment_19_dataset_comparison'),
    20: ('Phase Quantization',           'experiment_20_phase_quantization'),
}


def apply_quick_mode():
    """Reduce parameters for fast testing."""
    Config.FL_ROUNDS = 5
    Config.TRAIN_SAMPLES = 200
    Config.TEST_SAMPLES = 200
    Config.LOCAL_EPOCHS = 2
    print("[Quick mode] Reduced FL_ROUNDS=5, SAMPLES=200")


def run_experiments(exp_ids, results_dir=None):
    """Run a set of experiments by ID, returning timing info."""
    if results_dir:
        Config.RESULTS_DIR = results_dir

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    runner = AdvancedExperiments(Config)
    timings = {}
    status = {}

    print(f"\n{'='*60}")
    print(f"FL-RIS EXPERIMENT SUITE — {len(exp_ids)} experiments")
    print(f"Config: {Config.MODEL_TYPE} | {Config.NOC_TOPOLOGY} | "
          f"{Config.NOC_PROTOCOL} | {Config.AGGREGATION_METHOD}")
    print(f"Results: {Config.RESULTS_DIR}")
    print(f"{'='*60}")

    for eid in exp_ids:
        if eid not in EXPERIMENTS:
            print(f"\n[SKIP] Experiment {eid} not found")
            continue

        name, method_name = EXPERIMENTS[eid]
        print(f"\n{'─'*50}")
        print(f"Experiment {eid}: {name}")
        print(f"{'─'*50}")

        t0 = time.time()
        try:
            method = getattr(runner, method_name)
            method()
            elapsed = time.time() - t0
            timings[eid] = elapsed
            status[eid] = 'OK'
            print(f"  [OK] {name} — {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            timings[eid] = elapsed
            status[eid] = f'FAIL: {e}'
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for eid in exp_ids:
        if eid in status:
            name = EXPERIMENTS.get(eid, ('Unknown',))[0]
            t = timings.get(eid, 0)
            s = status[eid]
            print(f"  {eid:2d}. {name:30s} {t:8.1f}s  {s}")

    # Save timings
    timings_path = os.path.join(Config.RESULTS_DIR, 'advanced_experiments',
                                'experiment_timings.json')
    os.makedirs(os.path.dirname(timings_path), exist_ok=True)
    with open(timings_path, 'w') as f:
        json.dump({str(k): v for k, v in timings.items()}, f, indent=2)

    return timings, status


def run_campaign(exp_ids, seeds):
    """Run experiments across multiple seeds for statistical significance."""
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    campaign_dir = f"results/campaign_{timestamp}"
    os.makedirs(campaign_dir, exist_ok=True)

    all_timings = {}
    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"SEED = {seed}")
        print(f"{'#'*60}")

        Config.SEED = seed
        # Clear cached datasets for new seed
        cache_path = os.path.join(Config.DATA_DIR, 'datasets.pkl')
        if os.path.exists(cache_path):
            os.remove(cache_path)

        seed_dir = os.path.join(campaign_dir, f'seed_{seed}')
        timings, _ = run_experiments(exp_ids, results_dir=seed_dir)
        all_timings[seed] = timings

    # Save campaign summary
    with open(os.path.join(campaign_dir, 'campaign_summary.json'), 'w') as f:
        json.dump({'seeds': seeds, 'experiments': exp_ids,
                   'timings': {str(k): v for k, v in all_timings.items()}}, f, indent=2)

    print(f"\nCampaign saved to: {campaign_dir}")


def generate_report():
    """Generate report from existing results."""
    from utils.report_generator import ReportGenerator
    gen = ReportGenerator(Config.RESULTS_DIR, output_dir='report')
    gen.generate_full_report()
    print("Report generated in: report/")


def main():
    parser = argparse.ArgumentParser(description='FL-RIS Unified Experiment Runner')
    parser.add_argument('--exp', nargs='+', type=int,
                        help='Experiment IDs to run (1-20). Omit for all.')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: reduced rounds and samples')
    parser.add_argument('--campaign', action='store_true',
                        help='Multi-seed campaign mode')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[42, 123, 456, 789, 1024],
                        help='Seeds for campaign mode')
    parser.add_argument('--report-only', action='store_true',
                        help='Generate report from existing results')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Custom results directory')
    args = parser.parse_args()

    if args.quick:
        apply_quick_mode()

    if args.report_only:
        generate_report()
        return

    exp_ids = args.exp if args.exp else list(range(1, 21))

    if args.campaign:
        run_campaign(exp_ids, args.seeds)
    else:
        run_experiments(exp_ids, results_dir=args.results_dir)


if __name__ == '__main__':
    main()

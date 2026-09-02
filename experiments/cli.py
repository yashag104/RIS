"""Command-line entry points for the RIS experiment package."""

from config import Config

from .logging_utils import (
    configure_experiment_logging,
    get_experiment_logger,
    resolve_log_level,
)
from .suite import AdvancedExperiments

logger = get_experiment_logger("experiments.cli")

def run_all_experiments():
    """Run complete experiment suite"""
    configure_experiment_logging(
        f"{Config.RESULTS_DIR}/advanced_experiments",
        level=resolve_log_level(Config),
    )
    logger.info("\n" + "=" * 60)
    logger.info("ADVANCED EXPERIMENTS SUITE")
    logger.info("=" * 60)

    experiments = AdvancedExperiments(Config)

    all_results = {}

    # Run each experiment
    all_results['local_epochs'] = experiments.experiment_1_local_epochs_variation()
    all_results['quantization'] = experiments.experiment_2_quantization_levels()
    all_results['compression'] = experiments.experiment_3_model_compression()
    all_results['mobility'] = experiments.experiment_4_user_mobility()
    all_results['non_iid'] = experiments.experiment_5_non_iid_heterogeneity()
    all_results['pilots'] = experiments.experiment_6_pilot_overhead()
    all_results['noc'] = experiments.experiment_7_noc_traffic_vs_power()
    all_results['comparison'] = experiments.experiment_8_federated_vs_centralized()
    all_results['baselines'] = experiments.experiment_9_baseline_comparison()
    all_results['multiuser'] = experiments.experiment_10_multiuser_comparison()
    all_results['fl_algos'] = experiments.experiment_11_fl_algorithms()
    all_results['architectures'] = experiments.experiment_12_architectures()
    all_results['csi_robustness'] = experiments.experiment_13_csi_robustness()
    all_results['topology'] = experiments.experiment_14_topology_comparison()
    all_results['protocol'] = experiments.experiment_15_protocol_comparison()
    all_results['optimization'] = experiments.experiment_16_optimization_techniques()
    all_results['golden_ratio'] = experiments.experiment_17_tile_pixel_golden_ratio()
    all_results['duty_cycling'] = experiments.experiment_18_duty_cycling()
    all_results['datasets'] = experiments.experiment_19_dataset_comparison()
    all_results['phase_quantization'] = experiments.experiment_20_phase_quantization()

    logger.info("\n" + "=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE!")
    logger.info("=" * 60)

    return all_results


def run_new_experiments():
    """Run only the new experiments (9 and 10)"""
    configure_experiment_logging(
        f"{Config.RESULTS_DIR}/advanced_experiments",
        level=resolve_log_level(Config),
    )
    logger.info("\n" + "=" * 60)
    logger.info("NEW EXPERIMENTS: Baseline Comparison & Multi-User MIMO")
    logger.info("=" * 60)

    experiments = AdvancedExperiments(Config)

    results = {}
    results['baselines'] = experiments.experiment_9_baseline_comparison()
    results['multiuser'] = experiments.experiment_10_multiuser_comparison()

    logger.info("\n" + "=" * 60)
    logger.info("NEW EXPERIMENTS COMPLETE!")
    logger.info("=" * 60)

    return results


def run_journal_experiments():
    """Run only the journal-quality experiments (14-19)"""
    configure_experiment_logging(
        f"{Config.RESULTS_DIR}/advanced_experiments",
        level=resolve_log_level(Config),
    )
    logger.info("\n" + "=" * 60)
    logger.info("JOURNAL EXPERIMENTS: Topologies, Protocols, Optimization, Golden Ratio, Duty Cycling, Datasets")
    logger.info("=" * 60)

    experiments = AdvancedExperiments(Config)

    results = {}
    results['topology'] = experiments.experiment_14_topology_comparison()
    results['protocol'] = experiments.experiment_15_protocol_comparison()
    results['optimization'] = experiments.experiment_16_optimization_techniques()
    results['golden_ratio'] = experiments.experiment_17_tile_pixel_golden_ratio()
    results['duty_cycling'] = experiments.experiment_18_duty_cycling()
    results['datasets'] = experiments.experiment_19_dataset_comparison()
    results['phase_quantization'] = experiments.experiment_20_phase_quantization()

    logger.info("\n" + "=" * 60)
    logger.info("JOURNAL EXPERIMENTS COMPLETE!")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--new-only':
        run_new_experiments()
    elif len(sys.argv) > 1 and sys.argv[1] == '--journal':
        run_journal_experiments()
    else:
        run_all_experiments()

"""System-level evaluation across the whole RIS, not one tile at a time.

The per-tile evaluation in :meth:`src.client.RISClient.compute_snr_improvement`
scores a single ``ELEMENTS_PER_TILE``-element tile as though it were the entire
surface. With the default geometry that is 64 of the configured 1024 elements,
so ``TOTAL_RIS_ELEMENTS`` never influenced any reported number and the absolute
SNRs came out roughly 24 dB below what the described hardware would deliver.

This module evaluates the surface the config actually describes: every tile
predicts phases for its own elements from its own CSI, and all contributions are
summed coherently against one shared scene.

Requires datasets built with ``SHARED_SCENE_TILES`` (see
:func:`src.dataset_utils.create_system_test_dataset`) -- summing tiles drawn from
independent scenes is meaningless.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.channel_model import combine_tile_phases
from utils.logger import logger


def _dbm_to_watts(dbm: float) -> float:
    return 10 ** ((dbm - 30) / 10)


def evaluate_system(
    model,
    tile_datasets: list,
    config,
    num_samples: int = 100,
    device=None,
) -> dict[str, Any]:
    """Score the combined multi-tile surface.

    Args:
        model: Trained global model, shared by every tile.
        tile_datasets: Sample-aligned per-tile datasets (one scene per index).
        config: Config providing power, noise and interference settings.
        num_samples: Number of scenes to evaluate.

    Returns:
        Dict of system-level SNR and sum-rate metrics. ``*_per_tile`` fields
        report the single-tile equivalent so the gain from combining is visible.
    """
    if not tile_datasets:
        raise ValueError("tile_datasets is empty; nothing to evaluate")

    lengths = {len(d) for d in tile_datasets}
    if len(lengths) != 1:
        raise ValueError(
            f"tile datasets must be sample-aligned, got differing lengths {sorted(lengths)}"
        )

    device = device or getattr(config, 'DEVICE', torch.device('cpu'))
    model = model.to(device)
    model.eval()

    noise_power = _dbm_to_watts(config.NOISE_POWER_DBM)
    tx_power = _dbm_to_watts(config.TX_POWER_DBM)
    cross_talk = getattr(config, 'CROSS_TALK_FACTOR', 0.1)
    num_samples = min(num_samples, len(tile_datasets[0]))

    snr_no_ris, snr_system, snr_single_tile, snr_genie = [], [], [], []
    sum_rate_system, sum_rate_no_ris = [], []

    with torch.no_grad():
        for i in range(num_samples):
            # h_direct is shared across tiles by construction; take it from tile 0.
            h_direct = tile_datasets[0].metadata[i]['H_direct']

            cascades, phases = [], []
            for dataset in tile_datasets:
                md = dataset.metadata[i]
                cascade = md['H_ris'] * md['h_bs_ris']  # (U, N_tile)

                features, _ = dataset[i]
                predicted = model(features.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                # Same phase convention as the per-tile evaluation.
                applied = np.mod(predicted + md.get('phase_offset', 0.0), 2 * np.pi)

                cascades.append(cascade)
                phases.append(applied)

            total = combine_tile_phases(h_direct, cascades, phases)
            powers = tx_power * np.abs(total) ** 2
            direct_powers = tx_power * np.abs(h_direct) ** 2

            # One tile only, for the like-for-like comparison.
            single = combine_tile_phases(h_direct, cascades[:1], phases[:1])

            # Genie: every tile's elements aligned with the direct path per user.
            genie_total = np.asarray(h_direct, dtype=complex).copy()
            for cascade in cascades:
                genie_total = genie_total + np.sum(np.abs(cascade), axis=-1) * np.exp(
                    1j * np.angle(h_direct)
                )

            snr_no_ris.append(10 * np.log10(direct_powers[0] / noise_power))
            snr_system.append(10 * np.log10(powers[0] / noise_power))
            snr_single_tile.append(
                10 * np.log10(tx_power * np.abs(single[0]) ** 2 / noise_power)
            )
            snr_genie.append(
                10 * np.log10(tx_power * np.abs(genie_total[0]) ** 2 / noise_power)
            )

            sum_rate_system.append(_sum_rate(powers, noise_power, cross_talk))
            sum_rate_no_ris.append(_sum_rate(direct_powers, noise_power, cross_talk))

    total_elements = len(tile_datasets) * tile_datasets[0].num_ris_elements
    metrics = {
        'num_tiles': len(tile_datasets),
        'elements_per_tile': tile_datasets[0].num_ris_elements,
        'total_elements': total_elements,
        'snr_no_ris_mean': float(np.mean(snr_no_ris)),
        'snr_system_mean': float(np.mean(snr_system)),
        'snr_single_tile_mean': float(np.mean(snr_single_tile)),
        'snr_genie_mean': float(np.mean(snr_genie)),
        'system_gain_over_no_ris': float(np.mean(snr_system) - np.mean(snr_no_ris)),
        'system_gain_over_single_tile': float(
            np.mean(snr_system) - np.mean(snr_single_tile)
        ),
        'system_optimality_gap': float(np.mean(snr_genie) - np.mean(snr_system)),
        'sum_rate_system_mean': float(np.mean(sum_rate_system)),
        'sum_rate_no_ris_mean': float(np.mean(sum_rate_no_ris)),
        'sum_rate_gain': float(np.mean(sum_rate_system) - np.mean(sum_rate_no_ris)),
    }

    logger.info(
        f"[System] {metrics['num_tiles']} tiles x {metrics['elements_per_tile']} "
        f"= {total_elements} elements | "
        f"SNR {metrics['snr_system_mean']:.2f} dB "
        f"(+{metrics['system_gain_over_no_ris']:.2f} dB vs no-RIS, "
        f"+{metrics['system_gain_over_single_tile']:.2f} dB vs one tile, "
        f"gap {metrics['system_optimality_gap']:.2f} dB)"
    )
    return metrics


def _sum_rate(powers: np.ndarray, noise_power: float, cross_talk: float) -> float:
    """Sum-rate under the shared cross-talk interference model."""
    interference = cross_talk * (np.sum(powers) - powers)
    return float(np.sum(np.log2(1 + powers / (noise_power + interference))))

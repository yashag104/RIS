"""Physics and information-boundary regressions for the corrected study."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.link_level import (
    LinkLevelSuite,
    SceneSet,
    build_scene_set,
    design_genie,
    design_mrc_from_estimate,
    gains_from_phases,
)
from src.controlled_training import plateau, train_comparison
from src.pilot_probing import (
    LMMSEEstimator,
    pilot_features,
    probe_matrix,
)
from src.surface_channel import (
    SceneConfig,
    SurfaceGeometry,
    generate_surface_channels,
    surface_datasets,
)


def test_contiguous_aperture_and_tile_adjacency():
    g = SurfaceGeometry()
    p = g.element_positions()
    assert p.shape == (16, 64, 3)
    assert g.aperture_m == pytest.approx((.1714285714, .1714285714))
    assert np.linalg.norm(p[0, 7] - p[1, 0]) == pytest.approx(g.spacing)
    assert np.linalg.norm(p[0, 56] - p[4, 0]) == pytest.approx(g.spacing)
    assert np.unique(p.reshape(-1, 3), axis=0).shape[0] == 1024
    assert np.ptp(g.tile_centers()[:, 0]) < .13


def test_full_aperture_los_uses_spherical_distances():
    g = SurfaceGeometry(tile_rows=2, tile_cols=2)
    s = SceneConfig(k_factor_db=300)
    c = generate_surface_channels(3, g, s, np.random.default_rng(2))
    p = g.element_positions()
    path = np.linalg.norm(p - s.bs_position, axis=-1)[None] + np.linalg.norm(
        p[None] - c['user_pos'][:, None, None], axis=-1)
    expected = np.exp(-2j * np.pi * path / g.wavelength)
    assert np.allclose(c['cascade'] / np.abs(c['cascade']), expected, atol=2e-6)


def test_local_mrc_exact_optimum_and_locality():
    g = SurfaceGeometry(tile_rows=2, tile_cols=2)
    c = generate_surface_channels(9, g, SceneConfig(), np.random.default_rng(2))
    scenes = build_scene_set(surface_datasets(c, g))
    theta = design_mrc_from_estimate(scenes)
    bound = (np.abs(c['h_direct']) + np.abs(c['cascade']).sum((1, 2))) ** 2
    assert np.allclose(gains_from_phases(scenes, theta), bound, rtol=1e-6, atol=0)
    original = theta[:, 0].copy()
    scenes.cascade_est[:, 1:] *= np.exp(1.27j)
    assert np.array_equal(design_mrc_from_estimate(scenes)[:, 0], original)


def test_shared_direct_estimate_and_normalized_cascade_error():
    g = SurfaceGeometry(tile_rows=2, tile_cols=2)
    c = generate_surface_channels(300, g, SceneConfig(), np.random.default_rng(3))
    a = build_scene_set(surface_datasets(c, g, .1, 12))
    b = surface_datasets(c, g, .4, 12)
    assert all(np.array_equal(b[0].phase_offset, d.phase_offset) for d in b)
    b = build_scene_set(b)
    assert np.allclose(b.cascade_est - b.cascade_true, 2 * (a.cascade_est - a.cascade_true), atol=1e-15)
    nmse = np.mean(np.abs(a.cascade_est - a.cascade_true) ** 2) / np.mean(np.abs(a.cascade_true) ** 2)
    assert nmse == pytest.approx(.1, rel=.02)


def test_complex_covariance_has_correct_orientation():
    rng = np.random.default_rng(10)
    z = rng.normal(size=10000) + 1j * rng.normal(size=10000)
    vector = np.array([1, 1j, -.3 + .7j])
    x = z[:, None] * vector
    phases = np.array([[.3, -.8]])
    est = LMMSEEstimator(x, phases, shrinkage=0)
    test = x[:10] * (2 + 1j)
    y = test @ probe_matrix(phases).T
    assert np.allclose(est.estimate(y, 1e20), test, atol=1e-10)


def test_reflected_scaling_excludes_arbitrarily_strong_direct_path():
    c = np.ones((4, 4, 2), complex)
    hd = np.full(4, 1e6, complex)
    scenes = SceneSet(hd, hd, c, c, np.zeros((4, 4, 9), np.float32), np.zeros(4))
    cfg = SimpleNamespace(NOISE_POWER_DBM=-90, DEVICE='cpu', SEED=1)
    suite = LinkLevelSuite(cfg, {})
    theta = {k: None if k == 'no_ris' else design_genie(scenes) for k, _ in suite.schemes}
    result = suite.result_array_scaling(scenes, theta, 0)
    assert np.allclose(result['ideal_n_squared_db'], result['reflected_only_snr_db']['genie'])
    assert np.diff(result['ideal_n_squared_db'])[0] == pytest.approx(20*np.log10(2))
    assert result['mean_snr_db']['genie'][0] > 100


def test_pilot_inputs_use_only_observations_and_tile_coordinates():
    y = np.array([[1 + 2j, 3 - 4j], [.2j, 2j]])
    coords = np.array([[0, 0], [1, 1]])
    assert np.allclose(pilot_features(y, 10, coords), pilot_features(y * np.exp(.7j), 10, coords))
    assert pilot_features(y, 10, coords).shape == (2, 2, 7)


def test_pilot_training_uses_acquired_labels_and_no_oracle_offset():
    from run_pilot_limited import pilot_datasets
    from src.pilot_probing import ls_full_estimate, stack_channel
    g = SurfaceGeometry(tile_rows=1, tile_cols=2, pixel_rows=1, pixel_cols=2)
    c = generate_surface_channels(7, g, SceneConfig(), np.random.default_rng(11))
    x = stack_channel(c['h_direct'], c['cascade'])
    labels = ls_full_estimate(x, 1e10, np.random.default_rng(21))
    y = np.ones((7, 2), complex)
    ds = pilot_datasets(c, g, y, 1e10, labels)
    assert np.all(ds[0].phase_offset == 0)
    assert np.array_equal(ds[0].h_direct[:, 0], labels[:, 0])
    assert not np.array_equal(ds[0].h_direct[:, 0], c['h_direct'])
    c_other = dict(c, h_direct=c['h_direct'] * np.exp(.7j), cascade=c['cascade'] * 2)
    other = pilot_datasets(c_other, g, y, 1e10, labels)
    assert np.array_equal(ds[0].features, other[0].features)
    assert np.array_equal(ds[0].h_cascade, other[0].h_cascade)


def test_local_linear_probe_estimator_recovers_complex_mapping():
    from src.pilot_probing import LocalProbeRegressor
    rng = np.random.default_rng(2)
    y = rng.normal(size=(200, 4)) + 1j*rng.normal(size=(200, 4))
    weights = rng.normal(size=(4, 7)) + 1j*rng.normal(size=(4, 7))
    est = LocalProbeRegressor(y, y @ weights, ridge=0)
    assert np.allclose(est.estimate(y[-10:]), y[-10:] @ weights)


def test_plateau_is_not_decreasing_loss_or_budget_exhaustion():
    assert not plateau([.3, .2, .1], 2, 1e-4)
    assert plateau([.3, .1, .1, .1], 2, 1e-4)


def test_matched_optimizer_budgets_and_zero_round_local_baseline():
    from models.ris_net import MLPModel
    torch.set_num_threads(1)
    g = SurfaceGeometry(tile_rows=1, tile_cols=2, pixel_rows=1, pixel_cols=2)
    c = generate_surface_channels(5, g, SceneConfig(), np.random.default_rng(4))
    ds = surface_datasets(c, g)
    cfg = SimpleNamespace(DEVICE='cpu', MODEL_TYPE='MLP', LEARNING_RATE=.001, TX_POWER_DBM=30,
        NOISE_POWER_DBM=-90, LOCAL_EPOCHS=1, BATCH_SIZE=4, FL_ROUNDS=2, SEED=1,
        MIN_ROUNDS=2, PATIENCE=2, MIN_DELTA=1e-4)
    models, meta = train_comparison(lambda: MLPModel(9, 2, 8, 1), ds, ds, cfg)
    assert meta['fl_steps_per_client'] == 4
    assert meta['fl_total_optimizer_steps'] == meta['centralized_optimizer_steps'] == 8
    assert meta['centralized_client_budget_steps'] == 4
    assert meta['local_only_training_bytes'] == 0
    assert len(models['local_only']) == 2
    assert meta['stopping_reason'] == 'budget_exhausted'

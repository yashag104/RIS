"""Regression tests for the results-integrity bugs found in the Sept 2026 audit.

Each test here pins a specific way a published number turned out to be wrong.
They are deliberately about *measurement plumbing* rather than physics: the
failure mode in every case was a number that looked plausible, was plotted, and
was not actually measuring what its label claimed.

See test_physical_invariants.py and test_physics_regression.py for the physics.
"""

import inspect

import numpy as np
import pytest

from config import Config


# ---------------------------------------------------------------------------
# 1. CSI error must be relative to channel power, not an absolute variance.
# ---------------------------------------------------------------------------

def test_csi_error_scales_with_channel_power():
    """A given error_variance must degrade weak and strong channels equally.

    apply_csi_error previously added noise with an ABSOLUTE variance. Physical
    channel gains here are O(1e-4), so error_variance=0.01 buried every channel
    under ~1e6x its own power and the whole CSI-robustness sweep collapsed to
    "perfect CSI vs no CSI".
    """
    from src.channel_model import apply_csi_error

    rng = np.random.RandomState(0)
    base = (rng.randn(4096) + 1j * rng.randn(4096)) / np.sqrt(2)

    for scale in (1.0, 1e-4, 1e-8):
        channel = base * scale
        noisy = apply_csi_error(channel, error_variance=0.01)
        err_power = np.mean(np.abs(noisy - channel) ** 2)
        sig_power = np.mean(np.abs(channel) ** 2)
        # Error power should be ~1% of signal power regardless of scale.
        assert 0.005 < err_power / sig_power < 0.02, (
            f"CSI error not power-relative at channel scale {scale}: "
            f"ratio={err_power / sig_power:.4g}"
        )


def test_csi_error_phase_degrades_gradually():
    """Small CSI error must not produce a uniformly random phase estimate.

    Mean absolute phase error of ~90 deg means the estimate carries no
    information at all. Every non-zero variance used to hit exactly that.
    """
    from src.channel_model import apply_csi_error

    rng = np.random.RandomState(1)
    channel = ((rng.randn(4096) + 1j * rng.randn(4096)) / np.sqrt(2)) * 1e-4

    err_1pct = np.abs(np.angle(apply_csi_error(channel, 0.01) / channel))
    err_20pct = np.abs(np.angle(apply_csi_error(channel, 0.2) / channel))

    mean_1 = np.rad2deg(err_1pct.mean())
    mean_20 = np.rad2deg(err_20pct.mean())

    assert mean_1 < 10, f"1% CSI error should stay well under 10 deg, got {mean_1:.1f}"
    assert mean_1 < mean_20, "phase error must grow with error variance"
    assert mean_20 < 60, f"20% CSI error should not be near-random, got {mean_20:.1f}"


# ---------------------------------------------------------------------------
# 2. Fairness must be measured, never assigned from a closed form.
# ---------------------------------------------------------------------------

def test_fairness_index_is_not_hardcoded():
    """Experiment 5 must not compute fairness from alpha.

    It used to assign `fairness_index = 0.5 + alpha * 0.4`, a straight line that
    never touched the data and was published as a measurement.
    """
    from experiments import federated

    source = inspect.getsource(federated)
    offenders = [
        line.strip() for line in source.splitlines()
        if 'fairness_index' in line and '=' in line and 'alpha' in line
        and not line.strip().startswith('#')
    ]
    assert not offenders, (
        "fairness_index is being derived from alpha instead of measured: "
        f"{offenders}"
    )


def test_jain_index_matches_known_values():
    """Jain's index: 1.0 when equal, 1/n when all but one client score zero."""
    from utils.metrics import calculate_fairness_index

    assert calculate_fairness_index([0.8] * 8)['jains_index'] == pytest.approx(1.0)

    one_hot = calculate_fairness_index([1.0] + [0.0] * 7)['jains_index']
    assert one_hot == pytest.approx(1.0 / 8)

    mixed = calculate_fairness_index([0.9, 0.5, 0.3, 0.1])['jains_index']
    assert 1.0 / 4 < mixed < 1.0


# ---------------------------------------------------------------------------
# 3. Non-IID alpha must follow the Dirichlet convention and actually apply.
# ---------------------------------------------------------------------------

def test_non_iid_alpha_lowers_heterogeneity_as_it_grows():
    """Lower alpha must mean MORE heterogeneous, per Dirichlet convention.

    The shared-scene path ignored NON_IID_ALPHA entirely (so experiment 5 was
    five identical IID runs), and the legacy path used alpha directly as a bias
    multiplier, which ran the sweep backwards.
    """
    from src.dataset_utils import _spatial_scene_partition

    rng = np.random.RandomState(0)
    num_scenes, num_tiles = 400, 4
    room = (10.0, 10.0, 3.0)
    scenes = [
        {'user_positions': rng.uniform([0, 0, 0.5], [10, 10, 2.0], size=(1, 3))}
        for _ in range(num_scenes)
    ]
    tiles = [[2.0, 2.0, 1.5], [8.0, 2.0, 1.5], [8.0, 8.0, 1.5], [2.0, 8.0, 1.5]]

    def mean_distance_to_own_tile(alpha):
        np.random.seed(7)
        idx = _spatial_scene_partition(scenes, tiles, alpha=alpha, room_size=room)
        dists = []
        for t, tile in enumerate(tiles):
            pts = np.array([scenes[i]['user_positions'][0][:2] for i in idx[t]])
            dists.append(np.linalg.norm(pts - np.array(tile[:2]), axis=1).mean())
        return float(np.mean(dists))

    concentrated = mean_distance_to_own_tile(0.05)
    spread = mean_distance_to_own_tile(10.0)

    assert concentrated < spread, (
        "low alpha must concentrate each tile on nearby users (more non-IID); "
        f"got {concentrated:.2f} m at alpha=0.05 vs {spread:.2f} m at alpha=10"
    )


def test_non_iid_partition_rejects_invalid_alpha():
    from src.dataset_utils import _spatial_scene_partition

    scenes = [{'user_positions': np.zeros((1, 3))} for _ in range(10)]
    tiles = [[1.0, 1.0, 1.0]]
    for bad in (0.0, -1.0, float('inf')):
        with pytest.raises(ValueError):
            _spatial_scene_partition(scenes, tiles, alpha=bad, room_size=(10, 10, 3))


# ---------------------------------------------------------------------------
# 4. Every optimizer must use the same cascade convention.
# ---------------------------------------------------------------------------

def test_sdr_uses_same_cascade_as_every_other_method():
    """SDR must not conjugate the RIS-user channel.

    It was the only method that did, so it optimised a different channel from
    the one it was scored on and came out below random phases.
    """
    from baselines import sdr_optimizer

    source = inspect.getsource(sdr_optimizer)
    offenders = [
        line.strip() for line in source.splitlines()
        if 'conj(h_ris_user' in line.replace(' ', '').replace('np.', '')
        and not line.strip().startswith('#')
    ]
    assert not offenders, (
        f"SDR conjugates the RIS-user channel; the rest of the codebase does "
        f"not: {offenders}"
    )


def test_sdr_beats_random_phases_on_a_ris_dominated_channel():
    """With the direct link suppressed, SDR must clearly beat random phases."""
    cvxpy = pytest.importorskip("cvxpy")  # noqa: F841
    from baselines.sdr_optimizer import SDROptimizer

    rng = np.random.RandomState(3)
    N = 16
    noise = 1e-12
    sdr = SDROptimizer(num_elements=N, num_randomizations=60)

    wins = 0
    trials = 12
    for _ in range(trials):
        h_ris = (rng.randn(N) + 1j * rng.randn(N)) * 1e-3
        h_bs = (rng.randn(N) + 1j * rng.randn(N)) * 1e-3
        # Direct link far weaker than the cascade, so phases actually matter.
        h_direct = np.array([(rng.randn() + 1j * rng.randn()) * 1e-9])
        cascade = h_ris * h_bs

        res = sdr.optimize_phases(h_direct, h_ris, h_bs, noise)
        rand_phases = rng.uniform(0, 2 * np.pi, N)
        rand_db = 10 * np.log10(
            abs(h_direct[0] + np.dot(cascade, np.exp(1j * rand_phases))) ** 2 / noise
        )
        if res['snr_db'] > rand_db:
            wins += 1

    assert wins >= trials - 1, (
        f"SDR beat random phases in only {wins}/{trials} RIS-dominated trials"
    )


# ---------------------------------------------------------------------------
# 5. Topologies that differ physically must not produce identical numbers.
# ---------------------------------------------------------------------------

def test_folded_torus_differs_from_plain_torus():
    """Folding shortens the longest wire; the model must reflect that.

    Both topologies have the same hop counts, so a hop-only model reported
    byte-identical rows for them in the comparison table.
    """
    from src.noc_simulator import NoCSimulator

    torus = NoCSimulator(num_tiles=16, topology='Torus')
    folded = NoCSimulator(num_tiles=16, topology='FoldedTorus')

    assert folded.topology['max_link_length'] < torus.topology['max_link_length'], (
        "folded torus must have a shorter longest wire than a plain torus"
    )

    r_t = torus.simulate_fl_round(model_size_bytes=600_000, protocol='ParameterServer')
    r_f = folded.simulate_fl_round(model_size_bytes=600_000, protocol='ParameterServer')

    assert r_t['latency_us'] != r_f['latency_us'] or r_t['energy_nj'] != r_f['energy_nj'], (
        "Torus and FoldedTorus produce identical metrics; the model cannot "
        "distinguish two physically different fabrics"
    )


def test_topology_metrics_are_not_all_identical():
    """Distinct topologies must not collapse onto one latency number."""
    from src.noc_simulator import NoCSimulator

    energies = {}
    for name in ['Mesh', 'Torus', 'FoldedTorus', 'Tree', 'Butterfly', 'Ring']:
        sim = NoCSimulator(num_tiles=16, topology=name)
        r = sim.simulate_fl_round(model_size_bytes=600_000, protocol='ParameterServer')
        energies[name] = round(r['energy_nj'], 4)

    assert len(set(energies.values())) >= 5, (
        f"topologies are not being distinguished: {energies}"
    )


# ---------------------------------------------------------------------------
# 6. The 3GPP channel must block the direct link like the Rician one does.
# ---------------------------------------------------------------------------

def test_3gpp_applies_direct_link_blockage():
    """Without blockage the direct path dominates and the RIS is irrelevant.

    That is what made the 3GPP rows report a RIS gain of ~0.05 dB and 1-bit
    phase quantization appear to cost nothing.
    """
    from src.channel_model import ThreeGPPUMiChannel

    def mean_ris_gain_db(blockage_db, seed):
        np.random.seed(seed)
        cm = ThreeGPPUMiChannel(
            num_elements=64, frequency=28e9, direct_link_blockage_db=blockage_db
        )
        gains = []
        for _ in range(60):
            bs = np.array([5, 10, 2.5])
            user = np.random.uniform([0, 0, 0.5], [10, 10, 2], size=(1, 3))
            ris = np.array([5, 0, 1.5])
            ch = cm.generate_channel(bs, user, ris, 'LoS')
            h_d = ch['h_direct'][0]
            casc = ch['h_ris_user'][0] * ch['h_bs_ris']
            opt = np.mod(np.angle(h_d) - np.angle(casc), 2 * np.pi)
            with_ris = abs(h_d + np.sum(casc * np.exp(1j * opt))) ** 2
            gains.append(10 * np.log10(with_ris / abs(h_d) ** 2))
        return float(np.mean(gains))

    open_link = mean_ris_gain_db(0.0, seed=11)
    blocked = mean_ris_gain_db(30.0, seed=11)

    assert blocked > open_link + 1.0, (
        f"blockage has no effect on RIS gain: {open_link:.2f} dB open vs "
        f"{blocked:.2f} dB blocked"
    )


# ---------------------------------------------------------------------------
# 7. Provenance must record what actually ran.
# ---------------------------------------------------------------------------

def test_provenance_records_a_real_seed_attribute():
    """Provenance read a nonexistent RANDOM_SEED and always logged null."""
    import experiments.base as base

    source = inspect.getsource(base)
    assert "getattr(cfg, 'RANDOM_SEED'" not in source, (
        "provenance reads Config.RANDOM_SEED, which does not exist; "
        "Config defines SEED"
    )
    assert hasattr(Config, 'SEED'), "Config must define SEED"


def test_seed_experiment_publishes_active_seed():
    """The suite's per-experiment seed must reach the provenance block."""
    import run_all_experiments

    seed = run_all_experiments.seed_experiment(7)
    assert getattr(Config, '_ACTIVE_EXPERIMENT_SEED', None) == seed
    assert run_all_experiments.seed_experiment(7) == seed, "seeding must be deterministic"
    assert run_all_experiments.seed_experiment(8) != seed, "seeds must differ per experiment"

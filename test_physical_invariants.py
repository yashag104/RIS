"""Physical and mathematical invariants the results must satisfy.

Every check here corresponds to a defect that shipped in a previous set of
results. They are cheap and deterministic, and they fail loudly rather than
producing a plausible-looking but impossible number.

Run with:  python -m pytest test_physical_invariants.py -v
       or:  python test_physical_invariants.py
"""

import numpy as np
import torch

from config import Config
from src.channel_model import RicianChannel
from src.noc_simulator import NoCSimulator
from utils.metrics import dbm_to_watts

TOPOLOGIES = ['Mesh', 'Torus', 'FoldedTorus', 'Tree', 'Butterfly', 'Ring']
PROTOCOLS = ['ParameterServer', 'AllReduce', 'RingAllReduce', 'Gossip']


def test_noc_utilization_never_exceeds_one():
    """Link utilization is a busy fraction and cannot exceed 100%.

    A published figure once reported 146.44%, produced by dividing transmission
    time by an assumed 1 s round period and then clamping elsewhere.
    """
    for topo in TOPOLOGIES:
        for proto in PROTOCOLS:
            sim = NoCSimulator(num_tiles=16, topology=topo, bandwidth_gbps=10.0)
            m = sim.simulate_fl_round(606_750, proto)
            u = m['utilization']
            assert 0.0 <= u <= 1.0 + 1e-9, f"{topo}/{proto}: utilization {u}"


def test_noc_latency_and_energy_positive():
    """Latency and energy are positive and finite for every configuration."""
    for topo in TOPOLOGIES:
        for proto in PROTOCOLS:
            sim = NoCSimulator(num_tiles=16, topology=topo, bandwidth_gbps=10.0)
            m = sim.simulate_fl_round(606_750, proto)
            assert m['latency_us'] > 0 and np.isfinite(m['latency_us'])
            assert m['energy_j'] > 0 and np.isfinite(m['energy_j'])
            assert m['congestion_ratio'] >= 1.0 - 1e-9, (
                f"{topo}/{proto}: max link load below the mean is impossible")


def test_noc_topology_ordering():
    """Torus beats Mesh on parameter-server latency via its wrap-around links.

    This is the claim the paper cites Dally & Towles for; if the contention
    model stops reproducing it, the claim is no longer supported.
    """
    lat = {}
    for topo in ['Mesh', 'Torus']:
        sim = NoCSimulator(num_tiles=16, topology=topo, bandwidth_gbps=10.0)
        lat[topo] = sim.simulate_fl_round(606_750, 'ParameterServer')['latency_us']
    assert lat['Torus'] < lat['Mesh'], lat


def test_ring_allreduce_is_bandwidth_optimal():
    """RingAllReduce has the lowest aggregation latency of the four protocols."""
    sim = NoCSimulator(num_tiles=16, topology='Torus', bandwidth_gbps=10.0)
    lat = {p: sim.simulate_fl_round(606_750, p)['latency_us'] for p in PROTOCOLS}
    assert min(lat, key=lat.get) == 'RingAllReduce', lat


def _sample_channels(n=60, blockage_db=None):
    if blockage_db is None:
        blockage_db = Config.DIRECT_LINK_BLOCKAGE_DB
    cm = RicianChannel(num_elements=Config.ELEMENTS_PER_TILE,
                       k_factor_db=Config.RICIAN_K_FACTOR_DB,
                       frequency=Config.FREQUENCY,
                       direct_link_blockage_db=blockage_db)
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        ch = cm.generate_channel(
            np.array([5, 10, 2.5]),
            rng.uniform([0, 0, 0.5], [10, 10, 2], size=(1, 3)),
            np.array([5, 0, 1.5]), "LoS")
        out.append((ch['h_direct'][0], ch['h_ris_user'][0] * ch['h_bs_ris']))
    return out


def test_ris_has_measurable_effect():
    """Optimal phases must beat random phases by a usable margin.

    With an unobstructed direct link the cascade sits ~44 dB below it and the
    best achievable gain is under 0.2 dB, which is what made every experiment
    in a previous run agree to three decimal places.
    """
    noise = dbm_to_watts(Config.NOISE_POWER_DBM)
    rng = np.random.default_rng(1)
    gains = []
    for hd, a in _sample_channels():
        rnd = hd + np.dot(a, np.exp(1j * rng.uniform(0, 2 * np.pi, len(a))))
        opt = abs(hd) + np.sum(np.abs(a))
        gains.append(10 * np.log10(abs(opt) ** 2 / noise)
                     - 10 * np.log10(abs(rnd) ** 2 / noise))
    mean_gain = float(np.mean(gains))
    assert mean_gain > 1.0, (
        f"RIS gain over random phases is only {mean_gain:.2f} dB; "
        "no experiment built on this channel can show a meaningful effect")


def test_no_optimizer_beats_the_genie_bound():
    """Coherent alignment is the global optimum; nothing may exceed it."""
    from baselines.alternating_optimization import AlternatingOptimization
    from baselines.sca_optimizer import SCAOptimizer

    noise = dbm_to_watts(Config.NOISE_POWER_DBM)
    samples = _sample_channels(n=15)

    for hd, a in samples:
        ceiling = 10 * np.log10((abs(hd) + np.sum(np.abs(a))) ** 2 / noise)

        # The optimizers take the two channel hops separately; a = h_r * g, so
        # pass a as h_ris_user with a unit second hop.
        ones = np.ones(len(a), dtype=complex)
        for opt in (AlternatingOptimization(num_elements=len(a), max_iterations=200),
                    SCAOptimizer(num_elements=len(a))):
            if hasattr(opt, 'optimize_phases'):
                res = opt.optimize_phases(h_direct=hd, h_ris_user=np.conj(a),
                                          h_bs_ris=ones, noise_power=noise)
                snr = res['snr_db'] if isinstance(res, dict) else None
                if snr is None:
                    _phases, hist = res
                    snr = hist[-1]
                assert snr <= ceiling + 0.5, (
                    f"{type(opt).__name__} reported {snr:.2f} dB above the "
                    f"genie bound {ceiling:.2f} dB")


def test_phase_model_outputs_are_circular():
    """Models emit (cos, sin) pairs and angles inside (-pi, pi]."""
    from models.ris_net import create_model
    from src.dataset_utils import expected_feature_dim

    D = expected_feature_dim(Config.ELEMENTS_PER_TILE, Config.NUM_USERS)
    for arch in ['MLP', 'GNN', 'CNN_Attention', 'Transformer']:
        net = create_model(arch, D, Config.ELEMENTS_PER_TILE, config=Config)
        x = torch.randn(4, D)
        comp = net.forward_components(x)
        assert comp.shape == (4, Config.ELEMENTS_PER_TILE, 2), (arch, comp.shape)
        ang = net(x)
        assert ang.shape == (4, Config.ELEMENTS_PER_TILE)
        assert torch.all(ang >= -np.pi - 1e-5) and torch.all(ang <= np.pi + 1e-5)


def test_phase_label_is_learnable_from_features():
    """The target must be recoverable from the features by construction.

    The MRC optimum is angle(h_direct) - angle(cascade_n), and both terms appear
    in the feature vector. Extra users add features the label does not depend
    on, which is why NUM_USERS is 1 for the single-user task.

    The LABEL itself, however, carries only the per-element part
    ``-angle(cascade_n)``; the global ``angle(h_direct)`` term is a single
    per-sample constant held in ``metadata['phase_offset']`` and re-applied when
    the phases are configured. Splitting it out measurably helps, because the
    offset is computable exactly from CSI and making the network learn it can
    only add error -- measured on achieved SNR (600 samples, single user):

        label without offset : circular MSE 0.466, SNR 9.285 dB
        label with offset    : circular MSE 1.078, SNR 8.117 dB
        genie                :                     SNR 9.569 dB

    So the invariant to enforce is that label + phase_offset reconstructs the
    full MRC solution recoverable from the features, which is what this checks.
    """
    from src.dataset_utils import RISChannelDataset

    ds = RISChannelDataset(
        num_samples=40, num_ris_elements=Config.ELEMENTS_PER_TILE,
        num_users=1, room_size=Config.ROOM_SIZE, frequency=Config.FREQUENCY,
        k_factor_db=Config.RICIAN_K_FACTOR_DB,
        grid_rows=Config.PIXEL_GRID_ROWS, grid_cols=Config.PIXEL_GRID_COLS,
        direct_link_blockage_db=Config.DIRECT_LINK_BLOCKAGE_DB)

    N = Config.ELEMENTS_PER_TILE
    f, y = ds.features, ds.labels
    # Layout: 3 position values, then direct.real(1), cascade.real(N),
    # direct.imag(1), cascade.imag(N)
    d_re, c_re = f[:, 3], f[:, 4:4 + N]
    d_im, c_im = f[:, 4 + N], f[:, 5 + N:5 + 2 * N]
    recovered = np.mod(np.arctan2(d_im, d_re)[:, None]
                       - np.arctan2(c_im, c_re), 2 * np.pi)

    offset = np.array([m['phase_offset'] for m in ds.metadata])[:, None]
    applied = y + offset  # what actually gets configured on the surface

    err = np.abs(np.remainder(recovered - applied + np.pi, 2 * np.pi) - np.pi)
    assert np.rad2deg(np.mean(err)) < 1.0, (
        f"label + phase_offset does not reconstruct the MRC optimum recoverable "
        f"from the features: {np.rad2deg(np.mean(err)):.2f} deg")

    # The label on its own must NOT contain the global offset -- that is the
    # whole point of splitting it out. Guard against a silent revert.
    err_with_offset = np.abs(np.remainder(recovered - y + np.pi, 2 * np.pi) - np.pi)
    assert np.rad2deg(np.mean(err_with_offset)) > 1.0, (
        "label appears to already include the global MRC offset; "
        "phase_offset would then be applied twice at configuration time")


def test_duty_cycling_thresholds_are_distinguishable():
    """Different duty-cycling thresholds must select different element counts.

    Absolute dB thresholds against channels near -100 dB made every threshold
    setting collapse onto the same fallback, so three strategies reported
    bit-identical SNR and energy savings.
    """
    ratios = []
    for thresh_db in (-10, -20, -30):
        active = []
        for _hd, a in _sample_channels(n=40):
            p_db = 10 * np.log10(np.abs(a) ** 2 + 1e-20)
            active.append(np.mean(p_db > thresh_db + np.max(p_db)))
        ratios.append(float(np.mean(active)))
    assert len(set(np.round(ratios, 4))) == len(ratios), (
        f"thresholds produce identical activation ratios: {ratios}")
    assert ratios == sorted(ratios), f"looser threshold must activate more: {ratios}"


if __name__ == '__main__':
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)

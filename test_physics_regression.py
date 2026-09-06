"""End-to-end guards against the regressions found in the P0/P1 audit.

Each test here corresponds to a failure mode that was silently present and made
every downstream comparison meaningless:

* the RIS had no physical effect (genie-optimal gain was 0.70 dB, so no-RIS and
  random-RIS SNR agreed to the third decimal);
* the model could not learn (cascade features were crushed ~900x by a shared
  normalizer, and the label carried a random global phase offset), so a trained
  network scored no better than predicting a constant;
* duty cycling and sleep scheduling were dead code that no pipeline ever called.
"""

import numpy as np
import pytest
import torch
from torch import nn

from config import Config
from src.channel_model import generate_ris_channel_dataset

MIN_GENIE_GAIN_DB = 5.0


def _dataset(num_samples, blockage_db=None, element_gain=None, seed=0):
    np.random.seed(seed)
    return generate_ris_channel_dataset(
        num_samples=num_samples,
        num_ris_elements=Config.ELEMENTS_PER_TILE,
        num_users=Config.NUM_USERS,
        room_size=Config.ROOM_SIZE,
        frequency=Config.FREQUENCY,
        k_factor_db=Config.RICIAN_K_FACTOR_DB,
        scenario=Config.CHANNEL_SCENARIO,
        grid_rows=Config.PIXEL_GRID_ROWS,
        grid_cols=Config.PIXEL_GRID_COLS,
        direct_link_blockage_db=(
            Config.DIRECT_LINK_BLOCKAGE_DB if blockage_db is None else blockage_db
        ),
        element_gain_enabled=(
            Config.RIS_ELEMENT_GAIN_ENABLED if element_gain is None else element_gain
        ),
    )


def _snr_db(signal_amp):
    noise = 10 ** ((Config.NOISE_POWER_DBM - 30) / 10)
    tx = 10 ** ((Config.TX_POWER_DBM - 30) / 10)
    return 10 * np.log10(tx * np.abs(signal_amp) ** 2 / noise)


def _circular_mse(pred, target):
    return float(np.mean(np.angle(np.exp(1j * (pred - target))) ** 2))


def test_genie_optimal_gain_exceeds_floor():
    """The RIS must be able to move the received SNR by a meaningful margin.

    Without an obstructed direct link the cascaded path is ~3500x weaker per
    element and the genie-optimal gain collapses to ~0.7 dB, which makes every
    method comparison in the repo a comparison of rounding noise.
    """
    _, labels, metadata = _dataset(150)

    no_ris, genie = [], []
    for md, label in zip(metadata, labels):
        h_direct = md['H_direct'][0]
        h_cascade = md['H_ris'][0] * md['h_bs_ris']
        phases = np.mod(label + md['phase_offset'], 2 * np.pi)
        no_ris.append(_snr_db(h_direct))
        genie.append(_snr_db(h_direct + np.sum(h_cascade * np.exp(1j * phases))))

    gain = float(np.mean(genie) - np.mean(no_ris))
    assert gain > MIN_GENIE_GAIN_DB, (
        f"genie-optimal RIS gain {gain:.2f} dB <= {MIN_GENIE_GAIN_DB} dB floor; "
        "the RIS has no physical effect in this channel configuration"
    )


def test_stored_label_reproduces_genie_solution():
    """label + metadata['phase_offset'] must recover the true MRC optimum.

    The global offset is factored out of the label so the network has learnable
    structure; if it is not added back at application time the phases are wrong.
    """
    _, labels, metadata = _dataset(60, seed=1)

    for md, label in zip(metadata, labels):
        h_direct = md['H_direct'][0]
        h_cascade = md['H_ris'][0] * md['h_bs_ris']
        reconstructed = np.mod(label + md['phase_offset'], 2 * np.pi)
        true_optimal = np.mod(np.angle(h_direct) - np.angle(h_cascade), 2 * np.pi)
        assert _circular_mse(reconstructed, true_optimal) < 1e-6


def test_feature_blocks_are_comparably_scaled():
    """Direct and cascade feature blocks must not differ by orders of magnitude.

    A single shared RMS normalizer left the cascade block -- which carries
    essentially all the label information -- at RMS 0.0091 against 8.15 for the
    direct block.
    """
    features, _, _ = _dataset(120, seed=2)

    pos = Config.NUM_USERS * 3
    n_users, n_el = Config.NUM_USERS, Config.ELEMENTS_PER_TILE
    direct_rms = np.sqrt(np.mean(features[:, pos:pos + n_users] ** 2))
    cascade_rms = np.sqrt(np.mean(features[:, pos + n_users:pos + n_users + n_el] ** 2))

    ratio = max(direct_rms, cascade_rms) / max(min(direct_rms, cascade_rms), 1e-12)
    assert ratio < 10.0, (
        f"feature block RMS ratio {ratio:.1f} (direct {direct_rms:.4f}, "
        f"cascade {cascade_rms:.4f}); cascade features are being crushed"
    )


@pytest.mark.slow
def test_trained_model_beats_constant_baseline():
    """A trained network must beat the best constant prediction.

    Before the label and normalization fixes both scored ~3.2 circular MSE,
    which is what "the model isn't learning" actually looked like.
    """
    torch.manual_seed(0)
    features, labels, _ = _dataset(1200, seed=3)
    n_el = Config.ELEMENTS_PER_TILE
    split = 1000
    x_tr, y_tr = features[:split], labels[:split]
    x_te, y_te = features[split:], labels[split:]

    best_constant = min(
        _circular_mse(np.full_like(y_te, c), y_te)
        for c in np.linspace(0, 2 * np.pi, 100)
    )

    # Regress on (sin, cos) so the 2*pi wrap does not create false gradients.
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(np.stack([np.sin(y_tr), np.cos(y_tr)], -1), dtype=torch.float32)
    model = nn.Sequential(
        nn.Linear(features.shape[1], 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, n_el * 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(250):
        optimizer.zero_grad()
        loss = ((model(xt).view(-1, n_el, 2) - yt) ** 2).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out = model(torch.tensor(x_te, dtype=torch.float32)).view(-1, n_el, 2).numpy()
    model_mse = _circular_mse(np.arctan2(out[..., 0], out[..., 1]), y_te)

    assert model_mse < 0.8 * best_constant, (
        f"model circular MSE {model_mse:.4f} vs constant baseline "
        f"{best_constant:.4f}; the network is not learning the phase structure"
    )


def test_objective_recovers_genie_for_single_user():
    """The differentiable objective must be able to reach the genie optimum.

    Optimizing phases directly (no network in the way) on a single user is the
    one case where the MRC solution is exactly achievable, so anything short of
    the genie SNR means the objective or its gradient is wrong.
    """
    from src.objectives import SumRateLoss, received_power

    class _Cfg(Config):
        TEST_SAMPLES = 32
        NUM_USERS = 1

    from src.dataset_utils import create_test_dataset
    dataset = create_test_dataset(_Cfg)

    h_direct = torch.stack([dataset._as_real(dataset.h_direct[i]) for i in range(len(dataset))])
    h_cascade = torch.stack([dataset._as_real(dataset.h_cascade[i]) for i in range(len(dataset))])

    noise = 10 ** ((Config.NOISE_POWER_DBM - 30) / 10)
    tx = 10 ** ((Config.TX_POWER_DBM - 30) / 10)
    loss_fn = SumRateLoss(noise, tx, cross_talk=0.0)

    def snr(theta):
        power = tx * received_power(theta, h_direct, h_cascade)[:, 0]
        return (10 * torch.log10(power / noise)).mean().item()

    theta = torch.zeros(len(dataset), Config.ELEMENTS_PER_TILE, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=0.1)
    for _ in range(500):
        optimizer.zero_grad()
        loss_fn(theta, h_direct, h_cascade).backward()
        optimizer.step()

    genie = torch.tensor(np.stack([
        np.mod(np.angle(dataset.h_direct[i][0]) - np.angle(dataset.h_cascade[i][0]), 2 * np.pi)
        for i in range(len(dataset))
    ]), dtype=torch.float32)

    assert snr(theta) > snr(genie) - 0.5, (
        f"objective converged to {snr(theta):.2f} dB but genie is {snr(genie):.2f} dB; "
        "the differentiable SNR objective is not recovering the optimum"
    )


@pytest.mark.slow
def test_gnn_can_represent_per_element_phases():
    """The GNN must be able to fit per-element phase targets.

    It previously fused a single pooled context with a LEARNED-CONSTANT node
    embedding, so every node saw the identical sample-dependent term and the only
    per-node variation could not depend on the current sample. The target
    ``theta_n ~ -angle(h_cascade_n)`` varies per element AND per sample, so the
    model could only emit "global shift + fixed per-node constant" -- it plateaued
    near 80 degrees of phase error no matter the objective.

    This is a capacity test, not a sensitivity probe: a randomly initialized GAT
    stack diffuses a single-input perturbation across the whole grid, so
    forward-pass localization sits at chance either way. Whether the function is
    *representable* is the thing that actually differs, so we overfit a small
    batch with the per-element pathway enabled and with it zeroed out.
    """
    from models.ris_net import create_model
    from src.dataset_utils import create_test_dataset

    class _Cfg(Config):
        TEST_SAMPLES = 16
        NUM_USERS = 2

    dataset = create_test_dataset(_Cfg)
    x = torch.tensor(dataset.features, dtype=torch.float32)
    y = torch.tensor(dataset.labels, dtype=torch.float32)
    target = torch.stack([torch.sin(y), torch.cos(y)], -1)

    def overfit(disable_element_path: bool) -> float:
        torch.manual_seed(0)
        model = create_model(
            "GNN", input_dim=x.shape[1],
            num_elements=Config.ELEMENTS_PER_TILE, config=_Cfg,
        )
        if disable_element_path:
            with torch.no_grad():
                model.element_proj.weight.zero_()
                model.element_proj.bias.zero_()
            model.element_proj.weight.requires_grad_(False)
            model.element_proj.bias.requires_grad_(False)

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], 1e-3
        )
        for _ in range(600):
            optimizer.zero_grad()
            out = model(x)
            loss = ((torch.stack([torch.sin(out), torch.cos(out)], -1) - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            out = model(x)
        residual = torch.atan2(torch.sin(out - y), torch.cos(out - y))
        return float((residual ** 2).mean())

    without = overfit(disable_element_path=True)
    with_path = overfit(disable_element_path=False)

    assert with_path < 0.6 * without, (
        f"per-element pathway gave circular MSE {with_path:.4f} vs {without:.4f} "
        "without it; the GNN is not using per-element channel information"
    )


def test_client_does_not_change_dataset_arity():
    """Building a client must not alter what its dataset yields.

    The channel-carrying objectives need (features, labels, h_direct, h_cascade),
    but callers reuse the same dataset object to build evaluation loaders that
    unpack two values. An earlier version flipped a flag on the dataset itself,
    which broke `RISClient.evaluate` for every caller that did so.
    """
    from models.ris_net import create_model
    from src.client import RISClient
    from src.dataset_utils import create_test_dataset

    class _Cfg(Config):
        TEST_SAMPLES = 24
        NUM_USERS = 2
        TRAINING_OBJECTIVE = 'sumrate'
        BATCH_SIZE = 8

    dataset = create_test_dataset(_Cfg)
    assert len(dataset[0]) == 2

    model = create_model(
        _Cfg.MODEL_TYPE,
        input_dim=dataset.get_input_dim(),
        num_elements=Config.ELEMENTS_PER_TILE,
        config=_Cfg,
    )
    client = RISClient(client_id=0, model=model, dataset=dataset, config=_Cfg)

    assert len(dataset[0]) == 2, "client construction changed the dataset's arity"

    # The client's own training batches must still carry the channels.
    batch = next(iter(client.train_loader))
    assert len(batch) == 4, "channel-based objective did not receive channels"


def test_shared_scene_tiles_are_sample_aligned():
    """Tiles must share one scene per sample, or they cannot be summed.

    Coherently combining tiles drawn from independent scenes is meaningless; this
    alignment is what makes TOTAL_RIS_ELEMENTS real rather than decorative.
    """
    from src.dataset_utils import create_non_iid_datasets

    class _Cfg(Config):
        NUM_TILES = 4
        TRAIN_SAMPLES = 20
        NUM_USERS = 3
        SHARED_SCENE_TILES = True

    datasets, tile_positions = create_non_iid_datasets(_Cfg, _Cfg.NUM_TILES)

    assert len({len(d) for d in datasets}) == 1
    assert len({tuple(p) for p in tile_positions}) == _Cfg.NUM_TILES

    for i in range(len(datasets[0])):
        reference = datasets[0].metadata[i]['H_direct']
        for tile in datasets[1:]:
            assert np.allclose(reference, tile.metadata[i]['H_direct']), (
                f"sample {i}: tiles disagree on the direct channel, so they are "
                "not looking at the same scene"
            )

    # Different tiles must still see genuinely different cascade channels,
    # otherwise the tiles are redundant rather than spatially diverse.
    assert not np.allclose(
        datasets[0].metadata[0]['H_ris'], datasets[1].metadata[0]['H_ris']
    )


def test_combining_tiles_beats_a_single_tile():
    """Summing N tiles must beat one tile by roughly 20*log10(N) dB.

    This is the guard on TOTAL_RIS_ELEMENTS actually being exercised.
    """
    from src.channel_model import combine_tile_phases
    from src.dataset_utils import create_system_test_dataset

    class _Cfg(Config):
        NUM_TILES = 4
        TEST_SAMPLES = 25
        NUM_USERS = 2
        SHARED_SCENE_TILES = True

    np.random.seed(5)
    positions = [[2.0, 2.0, 1.5], [8.0, 2.0, 1.5], [8.0, 8.0, 1.5], [2.0, 8.0, 1.5]]
    tiles = create_system_test_dataset(_Cfg, positions)

    # Per-sample gains, summarized by the median. A ratio of means across samples
    # is dominated by whichever few scenes happen to have the most received
    # power and swings by several dB run to run; the per-sample median is stable
    # and is what "how much does adding tiles help" actually means.
    gains_db = []
    for i in range(len(tiles[0])):
        h_direct = tiles[0].metadata[i]['H_direct']
        cascades, phases = [], []
        for tile in tiles:
            md = tile.metadata[i]
            cascades.append(md['H_ris'] * md['h_bs_ris'])
            phases.append(np.mod(tile.labels[i] + md['phase_offset'], 2 * np.pi))

        single = np.abs(combine_tile_phases(h_direct, cascades[:1], phases[:1])[0]) ** 2
        combined = np.abs(combine_tile_phases(h_direct, cascades, phases)[0]) ** 2
        gains_db.append(10 * np.log10(combined / single))

    median_gain = float(np.median(gains_db))
    # Coherent addition of 4 tiles is worth 20*log10(4) = 12.0 dB in the limit
    # where the reflected path dominates; 6 dB leaves ample room for the direct
    # path diluting the ratio while still failing if tiles add incoherently.
    assert median_gain > 6.0, (
        f"combining {_Cfg.NUM_TILES} tiles gained a median of only "
        f"{median_gain:.2f} dB over one tile; tiles are not adding coherently"
    )


def test_multiuser_optimizer_actually_optimizes():
    """The classical multi-user optimizer must beat its own starting point.

    With a fixed learning rate its steps were ~1e-6 radians, because the sum-rate
    gradient scales with absolute received power (~1e-18 at these path losses).
    Over 300 iterations the objective moved 0.002081 -> 0.002085, i.e. the
    "optimized" phases it reported were the random vector it started from. Every
    multi-user number in experiment_10 was therefore a random-phase measurement.

    It must also beat a single-user MRC solution -- otherwise there is no reason
    to run a multi-user optimizer at all.
    """
    from experiments.baselines_multiuser import BaselineMultiuserExperimentsMixin as MU

    noise = 10 ** ((Config.NOISE_POWER_DBM - 30) / 10)
    tx = 10 ** ((Config.TX_POWER_DBM - 30) / 10)
    n_users, n_el = 4, Config.ELEMENTS_PER_TILE

    rng = np.random.default_rng(0)
    h_direct = (rng.normal(size=n_users) + 1j * rng.normal(size=n_users)) * 1e-8
    h_ris = (rng.normal(size=(n_users, n_el)) + 1j * rng.normal(size=(n_users, n_el))) * 1e-9

    obj = MU.__new__(MU)

    def sum_rate(phases):
        _, rates = MU._multiuser_rates(h_direct, h_ris, phases, n_users, noise, tx)
        return sum(rates)

    np.random.seed(0)
    start = sum_rate(np.random.uniform(0, 2 * np.pi, n_el))

    optimized = sum_rate(
        MU._optimize_multiuser_phases(obj, h_direct, h_ris, n_users, noise, tx)
    )
    # The optimizer initializes here, so it must at minimum not regress from it.
    single_user_mrc = sum_rate(
        np.mod(np.angle(h_direct[0]) - np.angle(h_ris[0]), 2 * np.pi)
    )

    assert optimized > 2 * start, (
        f"optimizer moved sum-rate {start:.6f} -> {optimized:.6f}; "
        "steps are too small to escape the initialization"
    )
    assert optimized >= single_user_mrc, (
        f"multi-user optimizer ({optimized:.6f}) regressed below its own "
        f"single-user MRC starting point ({single_user_mrc:.6f})"
    )


def test_genie_baseline_is_an_upper_bound():
    """A genie/oracle baseline must not be beatable by anything.

    experiment_9 computed its 'optimal' row by applying the dataset label
    directly. The label omits the global angle(h_direct) offset by design, so
    that produced a misaligned configuration which was not optimal -- both
    federated_ours (7.32 dB) and random_search (6.58 dB) scored above the
    supposed upper bound of 6.11 dB. Genie phases must be recomputed from the
    raw channels, which is convention-independent.
    """
    _, labels, metadata = _dataset(80, seed=7)

    label_only, recomputed = [], []
    for md, label in zip(metadata, labels):
        h_direct = md['H_direct'][0]
        h_cascade = md['H_ris'][0] * md['h_bs_ris']

        # What the buggy code did: label applied without the offset.
        label_only.append(_snr_db(h_direct + np.sum(h_cascade * np.exp(1j * label))))

        # What a genie actually is.
        true_opt = np.mod(np.angle(h_direct) - np.angle(h_cascade), 2 * np.pi)
        recomputed.append(_snr_db(h_direct + np.sum(h_cascade * np.exp(1j * true_opt))))

    assert np.mean(recomputed) > np.mean(label_only) + 0.5, (
        f"applying the raw label ({np.mean(label_only):.2f} dB) scores as well as "
        f"the recomputed genie ({np.mean(recomputed):.2f} dB); the genie baseline "
        "is not being computed from the raw channels"
    )

    # And the genie must dominate every sample, not just on average.
    for md, label in zip(metadata, labels):
        h_direct = md['H_direct'][0]
        h_cascade = md['H_ris'][0] * md['h_bs_ris']
        true_opt = np.mod(np.angle(h_direct) - np.angle(h_cascade), 2 * np.pi)
        genie = np.abs(h_direct + np.sum(h_cascade * np.exp(1j * true_opt)))
        random_phases = np.random.uniform(0, 2 * np.pi, len(h_cascade))
        got = np.abs(h_direct + np.sum(h_cascade * np.exp(1j * random_phases)))
        assert genie >= got - 1e-9, "a random configuration beat the genie bound"


def test_duty_cycle_metrics_respond_to_flag():
    """Toggling DUTY_CYCLE_ENABLED must actually change reported metrics.

    Both the mask and the metrics were previously unreachable: no call site
    existed anywhere in the pipeline.
    """
    from src.client import RISClient

    class _Cfg(Config):
        pass

    # 'topk' keeps exactly the strongest dc_min_active_ratio fraction, so the
    # expected ratio is deterministic. The 'threshold' strategy depends on the
    # CSI's dynamic range -- with unit-scale synthetic noise almost every pixel
    # sits within 20 dB of the peak and nothing is masked, which says nothing
    # about whether the mask is wired through to the metrics. Threshold
    # behaviour itself is covered by
    # test_physical_invariants.test_duty_cycling_thresholds_are_distinguishable.
    np.random.seed(11)
    csi = (np.random.randn(Config.ELEMENTS_PER_TILE)
           + 1j * np.random.randn(Config.ELEMENTS_PER_TILE))
    phases = np.random.uniform(0, 2 * np.pi, Config.ELEMENTS_PER_TILE)

    ratios = {}
    for enabled in (False, True):
        _Cfg.DUTY_CYCLE_ENABLED = enabled
        client = RISClient.__new__(RISClient)
        client.config = _Cfg
        client.num_pixels = Config.ELEMENTS_PER_TILE
        client.duty_cycle_enabled = enabled
        client.dc_strategy = 'topk'
        client.dc_threshold_db = getattr(Config, 'DUTY_CYCLE_THRESHOLD_DB', -20.0)
        client.dc_min_active_ratio = 0.25
        client.pixel_mask = np.ones(Config.ELEMENTS_PER_TILE, dtype=bool)
        client.dc_history = []
        client.active_power_pixel = getattr(Config, 'ACTIVE_POWER_PIXEL', 0.01)
        client.sleep_power_pixel = getattr(Config, 'SLEEP_POWER_PIXEL', 0.001)

        client.apply_duty_cycle_to_phases(phases, csi_vector=csi)
        ratios[enabled] = client.get_duty_cycle_metrics()['active_ratio']

    assert ratios[False] == pytest.approx(1.0)
    assert ratios[True] == pytest.approx(0.25), (
        f"topk duty cycling reported active_ratio {ratios[True]} instead of the "
        "configured 0.25; the mask is not reaching the metrics"
    )

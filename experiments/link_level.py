"""Link-level experiments: BER, outage, spectral efficiency and array scaling.

Experiments 1-20 report *received SNR* and system cost. This module adds the
physical-layer view a communications venue expects -- error-rate waterfalls,
outage curves, ergodic spectral efficiency and the array-gain scaling law --
for every phase-design scheme on one common set of channel realizations.

Everything here scores designs against the **true** channel while *building*
them from whatever CSI the scheme is entitled to see, so CSI error degrades a
scheme through its design and never through its scoring.

Phase convention matches :func:`src.channel_model.combine_tile_phases`:

    h_eff = h_direct + sum_t sum_n cascade[t, n] * exp(j * theta[t, n])
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from baselines.alternating_optimization import ProjectedGradientAscent
from baselines.sca_optimizer import SISOPhaseSurrogate
from src.channel_model import apply_phase_noise, quantize_phases
from src.link_metrics import (
    ber_curve,
    outage_curve,
    snr_at_target,
    spectral_efficiency_curve,
)

# Display order: worst to best, with the proposed scheme just before the bound
# so a reader's eye lands on the two together.
SCHEMES = [
    ("no_ris", "No RIS (blocked direct)"),
    ("random_ris", "Random RIS phases"),
    ("local_mrc", "Local MRC (same noisy CSI)"),
    ("ao", "Projected-gradient control"),
    ("sca", "Surrogate control"),
    ("centralized_dl", "Centralized DL"),
    ("centralized_client_budget", "Centralized DL (client-step budget)"),
    ("local_only", "Local model (no federation)"),
    ("fed_1round", "One-round FedAvg"),
    ("fed_5round", "Five-round FedAvg"),
    ("fed_ris", "Fed-RIS (proposed)"),
    ("genie", "Perfect-CSI MRC (bound)"),
]
SCHEME_LABELS = dict(SCHEMES)
LEARNED_SCHEMES = ("centralized_dl", "fed_ris", "centralized_client_budget",
                   "local_only", "fed_1round", "fed_5round")


@dataclass
class SceneSet:
    """Channel realizations shared by every scheme, plus the CSI each one sees.

    Shapes: ``S`` scenes, ``T`` tiles, ``N`` elements per tile.
    """

    h_direct_true: np.ndarray   # (S,)      complex, user 0
    h_direct_est: np.ndarray    # (S,)      complex
    cascade_true: np.ndarray    # (S, T, N) complex
    cascade_est: np.ndarray     # (S, T, N) complex
    features: np.ndarray        # (T, S, D) float32, network input per tile
    phase_offset: np.ndarray    # (S,)      angle of the estimated direct path

    @property
    def num_scenes(self) -> int:
        return self.cascade_true.shape[0]

    @property
    def num_tiles(self) -> int:
        return self.cascade_true.shape[1]

    @property
    def elements_per_tile(self) -> int:
        return self.cascade_true.shape[2]

    @property
    def total_elements(self) -> int:
        return self.num_tiles * self.elements_per_tile


def build_scene_set(tile_datasets: list, user: int = 0) -> SceneSet:
    """Collect the per-tile datasets of one shared scene into arrays.

    ``tile_datasets`` must be sample-aligned and built from a shared scene (see
    :func:`src.dataset_utils.create_system_test_dataset`); summing tiles drawn
    from independent scenes is meaningless.
    """
    lengths = {len(d) for d in tile_datasets}
    if len(lengths) != 1:
        raise ValueError(f"tile datasets must be sample-aligned, got {sorted(lengths)}")

    md0 = tile_datasets[0].metadata
    h_direct_true = np.array([m["H_direct"][user] for m in md0], dtype=complex)
    h_direct_est = np.array(
        [m.get("H_direct_est", m["H_direct"])[user] for m in md0], dtype=complex
    )

    cascade_true, cascade_est = [], []
    for ds in tile_datasets:
        cascade_true.append(
            np.array([m["H_ris"][user] * m["h_bs_ris"] for m in ds.metadata])
        )
        cascade_est.append(
            np.array(
                [
                    m.get("H_ris_est", m["H_ris"])[user] * m.get("h_bs_ris_est", m["h_bs_ris"])
                    for m in ds.metadata
                ]
            )
        )

    return SceneSet(
        h_direct_true=h_direct_true,
        h_direct_est=h_direct_est,
        cascade_true=np.stack(cascade_true, axis=1),
        cascade_est=np.stack(cascade_est, axis=1),
        features=np.stack([ds.features for ds in tile_datasets], axis=0),
        phase_offset=np.array([m.get("phase_offset", 0.0) for m in md0], dtype=float),
    )


# --------------------------------------------------------------------------
# Phase designs. Every one returns theta of shape (S, T, N), radians.
# --------------------------------------------------------------------------

def design_random(scenes: SceneSet, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(
        0.0, 2 * np.pi, size=(scenes.num_scenes, scenes.num_tiles, scenes.elements_per_tile)
    )


def design_genie(scenes: SceneSet) -> np.ndarray:
    """Optimal continuous phases from *true* CSI: rotate every reflected path
    onto the direct path. Upper bound for the single-user case."""
    return np.mod(
        np.angle(scenes.h_direct_true)[:, None, None] - np.angle(scenes.cascade_true),
        2 * np.pi,
    )


def design_mrc_from_estimate(scenes: SceneSet) -> np.ndarray:
    """The same closed form, driven by the estimated CSI a real receiver has."""
    return np.mod(
        np.angle(scenes.h_direct_est)[:, None, None] - np.angle(scenes.cascade_est),
        2 * np.pi,
    )


def design_learned(scenes: SceneSet, model, device, batch_size: int = 256) -> np.ndarray:
    """Per-tile network prediction plus the direct-path offset.

    The label the networks are trained on drops the global ``angle(h_direct)``
    term (see ``src.channel_model._channels_to_dataset``); it is known from CSI
    at application time and added back here.
    """
    models = model if isinstance(model, list) else [model] * scenes.num_tiles
    per_tile = []
    with torch.no_grad():
        for t in range(scenes.num_tiles):
            model = models[t].to(device).eval()
            feats = torch.from_numpy(scenes.features[t]).float()
            out = []
            for start in range(0, feats.shape[0], batch_size):
                chunk = feats[start:start + batch_size].to(device)
                out.append(model(chunk).cpu().numpy())
            per_tile.append(np.concatenate(out, axis=0))
    theta = np.stack(per_tile, axis=1)                      # (S, T, N)
    return np.mod(theta + scenes.phase_offset[:, None, None], 2 * np.pi)


def design_iterative(
    scenes: SceneSet,
    method: str,
    noise_power: float,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Run AO or SCA over the whole surface, one solve per scene.

    Both solvers take ``(h_direct, h_ris_user, h_bs_ris)`` and internally form a
    cascaded coefficient -- SCA as ``h_ris_user * h_bs_ris`` and AO as
    ``conj(h_ris_user) * h_bs_ris``. We already have the cascade, so the second
    factor is set to ones and the first to whichever of the cascade or its
    conjugate reproduces the cascade under that solver's own convention. This
    keeps one phase convention across the whole comparison.

    Every scene is solved. A phase vector is a function of that scene's channel,
    so designs cannot be shared between scenes the way a trained network's
    weights can -- reusing one would score a solver on phases computed for a
    different channel, which is just noise.

    Returns:
        ``(theta, mean_solve_time_s)`` with ``theta`` of shape (S, T, N).
    """
    rng = np.random.default_rng(seed)
    n_total = scenes.total_elements
    ones = np.ones(n_total, dtype=complex)

    if method == "sca":
        solver = SISOPhaseSurrogate(num_elements=n_total, max_iterations=50, verbose=False)
    elif method == "ao":
        solver = ProjectedGradientAscent(num_elements=n_total, max_iterations=200, verbose=False)
    else:
        raise ValueError(f"unknown iterative method {method!r}")

    solve_times = []
    solved = []
    for i in range(scenes.num_scenes):
        a_est = scenes.cascade_est[i].reshape(-1)             # (T*N,)
        h_d = np.array([scenes.h_direct_est[i]])
        init = rng.uniform(0.0, 2 * np.pi, n_total)
        t0 = time.perf_counter()
        if method == "sca":
            out = solver.optimize_phases(h_d, a_est, ones, noise_power, initial_phases=init)
            phases = np.asarray(out["phases"] if isinstance(out, dict) else out[0])
        else:
            phases, _ = solver.optimize_phases(
                h_d, np.conj(a_est), ones, noise_power, initial_phases=init
            )
            phases = np.asarray(phases)
        solve_times.append(time.perf_counter() - t0)
        solved.append(phases.reshape(scenes.num_tiles, scenes.elements_per_tile))

    return np.mod(np.stack(solved, axis=0), 2 * np.pi), float(np.mean(solve_times))


# --------------------------------------------------------------------------
# Hardware impairments and gain evaluation
# --------------------------------------------------------------------------

def apply_hardware(
    theta: np.ndarray, quant_bits: int = 0, phase_noise_std_deg: float = 0.0
) -> np.ndarray:
    """Quantize to ``2**quant_bits`` states and add RMS phase jitter."""
    out = theta
    if quant_bits and quant_bits > 0:
        out, _ = quantize_phases(out, quant_bits)
    if phase_noise_std_deg and phase_noise_std_deg > 0:
        out = apply_phase_noise(out, phase_noise_std_deg)
    return np.mod(out, 2 * np.pi)


def gains_from_phases(
    scenes: SceneSet, theta: np.ndarray | None, tiles: int | None = None
) -> np.ndarray:
    """Channel power gain ``|h_eff|^2`` per scene, scored on the true channel.

    ``theta=None`` means the surface is absent (direct link only).
    ``tiles`` restricts the surface to its first ``tiles`` tiles, for the
    array-gain scaling sweep.
    """
    h = scenes.h_direct_true.astype(complex).copy()
    if theta is not None:
        cascade = scenes.cascade_true
        if tiles is not None:
            cascade, theta = cascade[:, :tiles], theta[:, :tiles]
        h = h + (cascade * np.exp(1j * theta)).sum(axis=(1, 2))
    return np.abs(h) ** 2


# --------------------------------------------------------------------------
# The suite
# --------------------------------------------------------------------------

class LinkLevelSuite:
    """Produces the six link-level results, all from one set of scenes.

    Args:
        config: :class:`~config.Config`-like object.
        models: ``{'fed_ris': nn.Module, 'centralized_dl': nn.Module}``.
        scene_builder: Callable ``(csi_error_variance) -> list[RISChannelDataset]``
            returning sample-aligned per-tile datasets over the *same* true
            channels, with CSI estimates drawn at the requested quality. Used by
            the CSI-robustness sweep.
        logger: Optional logger.
    """

    #: Stated transmit-power budget; rho is internal computation only.
    TX_POWER_DBM = np.arange(-10.0, 31.0, 1.0)
    MODULATIONS = ("QPSK", "16QAM")
    OUTAGE_THRESHOLDS = (1.0, 2.0, 4.0)
    QUANT_BITS = (1, 2, 3, 0)          # 0 == continuous
    PHASE_NOISE_DEG = (0.0, 5.0, 15.0, 30.0)
    #: Normalized CSI error eps = sigma_e^2 / E[|h|^2], i.e. -30 dB to 0 dB.
    CSI_ERROR_VARIANCES = (0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 1.0)
    BER_TARGETS = (1e-3, 1e-4)

    def __init__(self, config, models: dict, scene_builder=None, logger=None):
        self.config = config
        self.RHO_DB = self.TX_POWER_DBM - config.NOISE_POWER_DBM
        self.models = models
        self.schemes = [(k, v) for k, v in SCHEMES
                        if k not in LEARNED_SCHEMES or k in models]
        self.scene_builder = scene_builder
        self.device = getattr(config, "DEVICE", torch.device("cpu"))
        self.noise_power = 10 ** ((config.NOISE_POWER_DBM - 30) / 10)
        self.seed = int(getattr(config, "SEED", 42))
        self.rng = np.random.default_rng(self.seed)
        if logger is None:
            from utils.logger import logger as _default
            logger = _default
        self.log = logger
        self._solve_times: dict[str, float] = {}

    # ---- designs -------------------------------------------------------

    def design(self, scheme: str, scenes: SceneSet) -> np.ndarray | None:
        """Phase design for one scheme, or ``None`` for the no-RIS case."""
        if scheme == "no_ris":
            return None
        if scheme == "random_ris":
            # Fixed seed, not the advancing one: random phases do not depend on
            # CSI, so this row must be flat across the CSI sweep. Redrawing per
            # call would put sampling noise where the physics has none.
            return design_random(scenes, np.random.default_rng(self.seed + 1))
        if scheme == "genie":
            return design_genie(scenes)
        if scheme == "local_mrc":
            return design_mrc_from_estimate(scenes)
        if scheme in LEARNED_SCHEMES:
            return design_learned(scenes, self.models[scheme], self.device)
        if scheme in ("ao", "sca"):
            theta, t_solve = design_iterative(
                scenes, scheme, self.noise_power,
                seed=self.seed + (2 if scheme == "ao" else 3),
            )
            self._solve_times[scheme] = t_solve
            return theta
        raise ValueError(f"unknown scheme {scheme!r}")

    def gains_all_schemes(self, scenes: SceneSet, thetas: dict | None = None) -> dict:
        """``{scheme: |h_eff|^2 per scene}`` for every scheme in :data:`SCHEMES`."""
        thetas = thetas if thetas is not None else {}
        gains = {}
        for key, label in self.schemes:
            theta = thetas.get(key) if key in thetas else self.design(key, scenes)
            thetas[key] = theta
            gains[key] = gains_from_phases(scenes, theta)
            self.log.info(
                f"  [{label}] mean received SNR at P_t={self.config.TX_POWER_DBM} dBm: "
                f"{10 * np.log10(np.mean(gains[key]) * 10 ** ((self.config.TX_POWER_DBM - 30) / 10) / self.noise_power):.2f} dB"
            )
        return gains

    # ---- result 1: BER vs SNR -----------------------------------------

    def result_ber_vs_snr(self, gains: dict) -> dict:
        rho_db = self.RHO_DB
        out = {"rho_db": rho_db.tolist(), "tx_power_dbm": self.TX_POWER_DBM.tolist(), "modulations": {}, "snr_at_target": {}}
        for mod in self.MODULATIONS:
            curves, targets = {}, {}
            for key, gain in gains.items():
                curve = ber_curve(gain, rho_db, mod)
                curves[key] = curve.tolist()
                targets[key] = {
                    f"{t:.0e}": snr_at_target(rho_db, curve, t) for t in self.BER_TARGETS
                }
            out["modulations"][mod] = curves
            out["snr_at_target"][mod] = targets
        return out

    # ---- result 2: spectral efficiency --------------------------------

    def result_spectral_efficiency(self, gains: dict) -> dict:
        rho_db = self.RHO_DB
        curves = {k: spectral_efficiency_curve(g, rho_db).tolist() for k, g in gains.items()}
        genie = np.asarray(curves["genie"])
        fraction_of_bound = {
            k: (np.asarray(v) / np.maximum(genie, 1e-12)).tolist() for k, v in curves.items()
        }
        return {
            "rho_db": rho_db.tolist(), "tx_power_dbm": self.TX_POWER_DBM.tolist(),
            "spectral_efficiency": curves,
            "fraction_of_bound": fraction_of_bound,
        }

    # ---- result 3: outage ---------------------------------------------

    def result_outage(self, gains: dict) -> dict:
        rho_db = self.RHO_DB
        out = {"rho_db": rho_db.tolist(), "tx_power_dbm": self.TX_POWER_DBM.tolist(), "thresholds_bps_hz": list(self.OUTAGE_THRESHOLDS),
               "curves": {}}
        for r_th in self.OUTAGE_THRESHOLDS:
            out["curves"][f"{r_th:g}"] = {
                k: outage_curve(g, rho_db, r_th).tolist() for k, g in gains.items()
            }
        return out

    # ---- result 4: hardware impairments -------------------------------

    def result_hardware_impairments(self, scenes: SceneSet, thetas: dict) -> dict:
        """Quantization and phase jitter applied to the proposed design.

        Also reports the genie design under the same impairment, so the loss can
        be separated into "what quantization costs any scheme" and "what the
        predictor costs on top of it".
        """
        rho_db = self.RHO_DB
        res = {"rho_db": rho_db.tolist(), "tx_power_dbm": self.TX_POWER_DBM.tolist(), "quantization": {}, "phase_noise": {}}

        for scheme in ("local_mrc", "fed_ris", "genie"):
            if scheme not in thetas:
                continue
            res["quantization"][scheme] = {}
            for bits in self.QUANT_BITS:
                theta_q = apply_hardware(thetas[scheme], quant_bits=bits)
                g = gains_from_phases(scenes, theta_q)
                label = "continuous" if bits == 0 else f"{bits}-bit"
                res["quantization"][scheme][label] = {
                    "ber_qpsk": ber_curve(g, rho_db, "QPSK").tolist(),
                    "spectral_efficiency": spectral_efficiency_curve(g, rho_db).tolist(),
                    "mean_gain_db": float(10 * np.log10(np.mean(g))),
                }

            res["phase_noise"][scheme] = {}
            for std in self.PHASE_NOISE_DEG:
                theta_n = apply_hardware(thetas[scheme], phase_noise_std_deg=std)
                g = gains_from_phases(scenes, theta_n)
                res["phase_noise"][scheme][f"{std:g}deg"] = {
                    "ber_qpsk": ber_curve(g, rho_db, "QPSK").tolist(),
                    "mean_gain_db": float(10 * np.log10(np.mean(g))),
                }
        return res

    # ---- result 5: CSI robustness -------------------------------------

    def result_csi_robustness(self, rho_db_operating: float) -> dict:
        """Degradation of every scheme as the CSI estimate gets worse.

        Requires ``scene_builder``: each variance draws a fresh estimate over
        the *same* true channels, so the curves isolate estimator quality.
        """
        if self.scene_builder is None:
            return {}

        rho_db = self.RHO_DB
        rho_lin = 10 ** (rho_db_operating / 10)
        res = {
            "rho_db": rho_db.tolist(), "tx_power_dbm": self.TX_POWER_DBM.tolist(),
            "operating_rho_db": rho_db_operating,
            "csi_error_variances": list(self.CSI_ERROR_VARIANCES),
            "ber_at_operating_point": {k: [] for k, _ in self.schemes},
            "spectral_efficiency_at_operating_point": {k: [] for k, _ in self.schemes},
            "mean_snr_db": {k: [] for k, _ in self.schemes},
            "fraction_of_oracle_power": {k: [] for k, _ in self.schemes},
            "gap_to_local_mrc_db": {k: [] for k, _ in self.schemes},
            "ber_curves": {},
        }

        for var in self.CSI_ERROR_VARIANCES:
            self.log.info(f"[LinkLevel] CSI robustness: sigma_e^2 = {var}")
            scenes = build_scene_set(self.scene_builder(var))
            gains = self.gains_all_schemes(scenes)
            res["ber_curves"][f"{var:g}"] = {}
            for key, g in gains.items():
                gamma = rho_lin * g
                from src.link_metrics import awgn_ber
                res["ber_at_operating_point"][key].append(float(np.mean(awgn_ber(gamma, "QPSK"))))
                res["spectral_efficiency_at_operating_point"][key].append(
                    float(np.mean(np.log2(1 + gamma)))
                )
                res["mean_snr_db"][key].append(float(10 * np.log10(np.mean(gamma))))
                res["fraction_of_oracle_power"][key].append(float(np.mean(g / gains["genie"])))
                res["gap_to_local_mrc_db"][key].append(
                    float(10 * np.log10(np.mean(g) / np.mean(gains["local_mrc"]))))
                res["ber_curves"][f"{var:g}"][key] = ber_curve(g, rho_db, "QPSK").tolist()
        return res

    # ---- result 6: array-gain scaling ---------------------------------

    def result_array_scaling(
        self, scenes: SceneSet, thetas: dict, rho_db_operating: float
    ) -> dict:
        """Received SNR and spectral efficiency against the active surface size.

        The theoretical single-user law is a power gain of ``N^2`` for coherent
        combining (``20 log10 N``); the measured slope says how much of it each
        design actually realizes.
        """
        rho_lin = 10 ** (rho_db_operating / 10)
        tile_counts = [t for t in (1, 2, 4, 8, 12, 16) if t <= scenes.num_tiles]
        if scenes.num_tiles not in tile_counts:
            tile_counts.append(scenes.num_tiles)

        res = {
            "operating_rho_db": rho_db_operating,
            "tile_counts": tile_counts,
            "element_counts": [t * scenes.elements_per_tile for t in tile_counts],
            "mean_snr_db": {}, "spectral_efficiency": {}, "ber_qpsk": {},
            "snr_cdf": {},
        }
        from src.link_metrics import awgn_ber

        res["reflected_only_snr_db"] = {}
        for key, _label in self.schemes:
            theta = thetas[key]
            snrs, ses, bers = [], [], []
            for t in tile_counts:
                g = gains_from_phases(scenes, theta, tiles=t)
                gamma = rho_lin * g
                snrs.append(float(10 * np.log10(np.mean(gamma))))
                ses.append(float(np.mean(np.log2(1 + gamma))))
                bers.append(float(np.mean(awgn_ber(gamma, "QPSK"))))
            res["mean_snr_db"][key] = snrs
            res["spectral_efficiency"][key] = ses
            res["ber_qpsk"][key] = bers
            gamma_full = rho_lin * gains_from_phases(scenes, theta)
            res["snr_cdf"][key] = np.sort(10 * np.log10(gamma_full)).tolist()
            if theta is not None:
                res["reflected_only_snr_db"][key] = [float(10 * np.log10(rho_lin *
                    np.mean(np.abs((scenes.cascade_true[:, :t] *
                                    np.exp(1j * theta[:, :t])).sum((1, 2))) ** 2)))
                    for t in tile_counts]

        # Compare the N^2 law with reflected-only power. No choice of anchor
        # removes direct-path contamination from a total-power curve.
        genie = res["reflected_only_snr_db"]["genie"]
        anchor = 0
        res["ideal_anchor_index"] = anchor
        res["ideal_anchor_elements"] = res["element_counts"][anchor]
        res["ideal_n_squared_db"] = [
            genie[anchor] + 20 * np.log10(t / tile_counts[anchor]) for t in tile_counts
        ]
        res["ideal_reference_quantity"] = "reflected-only power; direct path excluded"
        amplitude = np.abs(scenes.cascade_true).sum((1, 2)) / scenes.total_elements
        res["uniform_amplitude_total_snr_db"] = [float(10 * np.log10(rho_lin *
            np.mean((np.abs(scenes.h_direct_true) + n * amplitude) ** 2)))
            for n in res["element_counts"]]
        return res

    # ---- driver --------------------------------------------------------

    def run(self, scenes: SceneSet) -> dict:
        self.log.info("[LinkLevel] designing phases for every scheme ...")
        thetas: dict = {}
        gains = self.gains_all_schemes(scenes, thetas)

        results = {
            "meta": {
                "num_scenes": scenes.num_scenes,
                "num_tiles": scenes.num_tiles,
                "elements_per_tile": scenes.elements_per_tile,
                "total_elements": scenes.total_elements,
                "noise_power_dbm": self.config.NOISE_POWER_DBM,
                "tx_power_dbm": self.config.TX_POWER_DBM,
                "frequency_hz": self.config.FREQUENCY,
                "direct_link_blockage_db": getattr(self.config, "DIRECT_LINK_BLOCKAGE_DB", 30.0),
                "scheme_labels": dict(self.schemes),
                "mean_solve_time_s": dict(self._solve_times),
            },
            "mean_gain_db": {k: float(10 * np.log10(np.mean(v))) for k, v in gains.items()},
        }

        self.log.info("[LinkLevel] R1 BER vs SNR")
        results["ber_vs_snr"] = self.result_ber_vs_snr(gains)

        # Predeclared physical operating point, independent of a model's result.
        rho_op = float(self.config.TX_POWER_DBM - self.config.NOISE_POWER_DBM)
        results["meta"]["operating_rho_db"] = rho_op
        self.log.info(f"[LinkLevel] operating point Pt = {self.config.TX_POWER_DBM:g} dBm")

        self.log.info("[LinkLevel] R2 spectral efficiency")
        results["spectral_efficiency"] = self.result_spectral_efficiency(gains)
        self.log.info("[LinkLevel] R3 outage probability")
        results["outage"] = self.result_outage(gains)
        self.log.info("[LinkLevel] R4 hardware impairments")
        results["hardware_impairments"] = self.result_hardware_impairments(scenes, thetas)
        self.log.info("[LinkLevel] R6 array-gain scaling")
        results["array_scaling"] = self.result_array_scaling(scenes, thetas, rho_op)
        self.log.info("[LinkLevel] R5 CSI robustness")
        results["csi_robustness"] = self.result_csi_robustness(rho_op)
        return results

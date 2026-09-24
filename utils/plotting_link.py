"""The six link-level paper figures.

One function per figure, plus :func:`render_all`. Everything is driven from the
JSON that ``run_link_level.py`` writes, so figures can be re-rendered without
re-running any physics.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.plotting import COLORS, IEEE_RC, _style_legend

plt.rcParams.update(IEEE_RC)


def _save(fig, save_path, name):
    """Save without the legacy live-Config stamp; render_all records hashes."""
    os.makedirs(save_path, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(Path(save_path)/f"{name}.{extension}", format=extension)
    plt.close(fig)

# One visual identity per scheme, shared by every figure so a reader can track a
# curve across the whole figure set. The proposed scheme is the only thick solid
# line; the bound is the only dotted black one.
STYLE = {
    "no_ris":         {"color": COLORS["gray"],   "marker": "x", "ls": ":",  "lw": 1.2, "label": "No RIS (blocked)"},
    "random_ris":     {"color": COLORS["brown"],  "marker": "v", "ls": "-.", "lw": 1.2, "label": "Random phases"},
    "ao":             {"color": COLORS["green"],  "marker": "^", "ls": "--", "lw": 1.3, "label": "Projected gradient"},
    "sca":            {"color": COLORS["orange"], "marker": "D", "ls": "--", "lw": 1.3, "label": "SISO surrogate"},
    "centralized_dl": {"color": COLORS["purple"], "marker": "s", "ls": "-",  "lw": 1.3, "label": "Centralized DL"},
    "fed_ris":        {"color": COLORS["red"],    "marker": "o", "ls": "-",  "lw": 2.0, "label": "Fed-RIS (proposed)"},
    "genie":          {"color": "black",          "marker": "",  "ls": ":",  "lw": 1.4, "label": "Perfect-CSI MRC (bound)"},
}
STYLE.update({
    "local_mrc": {"color": "#0072B2", "marker": "+", "ls": "--", "lw": 1.4, "label": "Local noisy-CSI MRC"},
    "local_only": {"color": "#009E73", "marker": "<", "ls": "-.", "lw": 1.2, "label": "Local model"},
    "fed_1round": {"color": "#CC79A7", "marker": "1", "ls": ":", "lw": 1.2, "label": "One-round FL"},
    "fed_5round": {"color": "#56B4E9", "marker": "2", "ls": ":", "lw": 1.2, "label": "Five-round FL"},
    "centralized_client_budget": {"color": "#777777", "marker": ">", "ls": "--", "lw": 1.2,
                                       "label": "Central (client steps)"},
})
ORDER = ["no_ris", "random_ris", "local_mrc", "ao", "sca", "centralized_dl",
         "centralized_client_budget", "local_only", "fed_1round", "fed_5round", "fed_ris", "genie"]
BER_FLOOR = 1e-7


def _plot_scheme(ax, x, y, key, markevery=6, mask_floor=False, **over):
    """Draw one scheme's curve in its fixed house style.

    ``mask_floor`` hides points that were clipped at the BER floor, so a
    waterfall ends where the simulation stops resolving it instead of running
    along the bottom of the axes as a flat line that means nothing.
    """
    st = dict(STYLE[key])
    st.update(over)
    label = st.pop("label")
    y = np.asarray(y, dtype=float)
    if mask_floor:
        y = np.where(y <= BER_FLOOR * 1.000001, np.nan, y)
    return ax.plot(x, y, label=label, markevery=markevery, markersize=4,
                   zorder=4 if key == "fed_ris" else 3, **st)


def _shared_legend(fig, axes, ncol=4, y=-0.02):
    """One legend under the whole figure, so no panel has to give up space."""
    handles, labels = [], []
    for ax in np.atleast_1d(axes).ravel():
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in labels:
                handles.append(h)
                labels.append(lbl)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol, frameon=False, fontsize=6.8, handlelength=2.0,
               columnspacing=1.4)


def _log_ber_axis(ax, floor=BER_FLOOR):
    ax.set_yscale("log")
    ax.set_ylim(floor * 2, 1.0)
    ax.grid(True, which="both", ls="--", lw=0.35, alpha=0.3)


# ==========================================================================
# Figure 1 — BER vs transmit SNR
# ==========================================================================

def plot_ber_vs_snr(results, save_path):
    blk = results["ber_vs_snr"]
    rho = np.array(blk["rho_db"]) + results["meta"]["noise_power_dbm"]
    mods = [m for m in ("QPSK", "16QAM") if m in blk["modulations"]]

    fig, axes = plt.subplots(1, len(mods), figsize=(7.16, 3.2), sharey=True)
    axes = np.atleast_1d(axes)

    for i, (ax, mod) in enumerate(zip(axes, mods)):
        curves = blk["modulations"][mod]
        for key in ORDER:
            if key in curves:
                _plot_scheme(ax, rho, curves[key], key, mask_floor=True)
        ax.axhline(1e-3, color="#999999", lw=0.6, ls="-", zorder=1)
        ax.text(rho[1], 1.3e-3, r"BER $=10^{-3}$", fontsize=6, color="#777777")
        _log_ber_axis(ax)
        ax.set_xlabel(r"Transmit power $P_t$ (dBm)")
        ax.set_title(f"({'ab'[i]}) {mod}", fontsize=9, loc="left")


    axes[0].set_ylabel("Bit error rate")
    _shared_legend(fig, axes, ncol=4, y=0.005)
    _save(fig, save_path, "fig1_ber_vs_snr")


# ==========================================================================
# Figure 2 — Ergodic spectral efficiency
# ==========================================================================

def plot_spectral_efficiency(results, save_path):
    blk = results["spectral_efficiency"]
    rho = np.array(blk["rho_db"]) + results["meta"]["noise_power_dbm"]
    se = blk["spectral_efficiency"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 3.2))

    for key in ORDER:
        if key in se:
            _plot_scheme(a1, rho, np.array(se[key]), key)
    a1.set_xlabel(r"Transmit power $P_t$ (dBm)")
    a1.set_ylabel("Ergodic spectral efficiency (bit/s/Hz)")
    a1.set_title("(a) Achievable rate", fontsize=9, loc="left")


    # (b) horizontal SNR gain over the no-RIS link at a fixed target rate.
    target = 4.0
    no_ris = np.array(se["no_ris"])
    gains, labels, colors = [], [], []
    for key in ORDER:
        if key == "no_ris" or key not in se:
            continue
        y = np.array(se[key])
        rho_here = np.interp(target, y, rho, left=np.nan, right=np.nan)
        rho_ref = np.interp(target, no_ris, rho, left=np.nan, right=np.nan)
        gains.append(rho_ref - rho_here)
        labels.append(STYLE[key]["label"].replace(" (proposed)", "\n(proposed)"))
        colors.append(STYLE[key]["color"])

    ypos = np.arange(len(labels))
    bars = a2.barh(ypos, gains, color=colors, edgecolor="black", linewidth=0.5, height=0.6)
    a2.set_yticks(ypos)
    a2.set_yticklabels(labels, fontsize=6.5)
    a2.invert_yaxis()
    finite = [g for g in gains if np.isfinite(g)]
    if finite:
        a2.set_xlim(0, max(finite) * 1.14)
    a2.set_xlabel(f"Transmit-power saving at {target:g} bit/s/Hz (dB)")
    a2.set_title("(b) Gain over the blocked direct link", fontsize=9, loc="left")
    if not finite:
        a2.text(0.5, 0.5, "Target not reached by both links\nwithin the transmit-power sweep",
                transform=a2.transAxes, ha="center", va="center", fontsize=7)
    for b, g in zip(bars, gains):
        if np.isfinite(g):
            a2.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2,
                    f"{g:.1f}", va="center", fontsize=6.5)
    a2.grid(True, axis="x", ls="--", lw=0.35, alpha=0.3)
    if not finite:
        a2.set_axis_off()

    _shared_legend(fig, [a1], ncol=4, y=0.005)
    _save(fig, save_path, "fig2_spectral_efficiency")


# ==========================================================================
# Figure 3 — Outage probability
# ==========================================================================

def plot_outage(results, save_path):
    blk = results["outage"]
    rho = np.array(blk["rho_db"]) + results["meta"]["noise_power_dbm"]
    thresholds = [f"{t:g}" for t in blk["thresholds_bps_hz"]]
    show = thresholds[:2] if len(thresholds) >= 2 else thresholds

    fig, axes = plt.subplots(1, len(show), figsize=(7.16, 3.2), sharey=True)
    axes = np.atleast_1d(axes)

    floor = 1.0 / max(results["meta"]["num_scenes"], 1)
    for i, (ax, th) in enumerate(zip(axes, show)):
        curves = blk["curves"][th]
        for key in ORDER:
            if key in curves:
                # Zero outages over S realizations means "below 1/S", not zero;
                # drawing them along the bottom would claim a resolution the
                # simulation does not have.
                y = np.where(np.array(curves[key]) <= 0, np.nan, curves[key])
                _plot_scheme(ax, rho, y, key)
        ax.axhline(floor, color="#bbbbbb", lw=0.6, ls="-", zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(floor * 0.6, 1.3)
        ax.grid(True, which="both", ls="--", lw=0.35, alpha=0.3)
        ax.set_xlabel(r"Transmit power $P_t$ (dBm)")
        ax.set_title(f"({'ab'[i]}) $R_{{th}} = {th}$ bit/s/Hz", fontsize=9, loc="left")


    axes[0].set_ylabel("Outage probability")
    axes[-1].text(
        0.98, 0.03,
        f"resolution limit = 1/{results['meta']['num_scenes']} realizations",
        transform=axes[-1].transAxes, ha="right", va="bottom", fontsize=5.5,
        color="#777777",
    )
    _shared_legend(fig, axes, ncol=4, y=0.005)
    _save(fig, save_path, "fig3_outage_probability")


# ==========================================================================
# Figure 4 — Hardware impairments
# ==========================================================================

def plot_hardware_impairments(results, save_path):
    blk = results["hardware_impairments"]
    rho = np.array(blk["rho_db"]) + results["meta"]["noise_power_dbm"]
    scheme = "fed_ris" if "fed_ris" in blk["quantization"] else "local_mrc"
    quant = blk["quantization"][scheme]
    quant_bound = blk["quantization"]["genie"]
    noise = blk["phase_noise"][scheme]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.16, 2.9))

    qorder = [k for k in ("1-bit", "2-bit", "3-bit", "continuous") if k in quant]
    qcolors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]
    for k, c in zip(qorder, qcolors):
        a1.plot(rho, np.where(np.array(quant[k]["ber_qpsk"]) <= BER_FLOOR * 1.000001,
                              np.nan, quant[k]["ber_qpsk"]), color=c, lw=1.4,
                marker="o", markevery=5, markersize=3.5, label=k)
    _log_ber_axis(a1)
    a1.set_xlabel(r"Transmit power $P_t$ (dBm)")
    a1.set_ylabel("BER (QPSK)")
    a1.set_title("(a) Phase quantization", fontsize=9, loc="left")
    _style_legend(a1, loc="lower left", fontsize=6.2)

    norder = sorted(noise, key=lambda s: float(s.replace("deg", "")))
    for k, c in zip(norder, [COLORS["red"], COLORS["orange"], COLORS["green"], COLORS["blue"]]):
        a2.plot(rho, np.where(np.array(noise[k]["ber_qpsk"]) <= BER_FLOOR * 1.000001,
                              np.nan, noise[k]["ber_qpsk"]), color=c, lw=1.4,
                marker="s", markevery=5, markersize=3.5,
                label=rf"$\sigma_\phi$ = {k.replace('deg', '')}$^\circ$")
    _log_ber_axis(a2)
    a2.set_xlabel(r"Transmit power $P_t$ (dBm)")
    a2.set_title("(b) RIS phase jitter", fontsize=9, loc="left")
    _style_legend(a2, loc="lower left", fontsize=6.2)

    # (c) SNR cost of quantization, measured vs the classic sinc bound.
    bits = [1, 2, 3]
    ref = quant["continuous"]["mean_gain_db"]
    ref_b = quant_bound["continuous"]["mean_gain_db"]
    meas = [quant[f"{b}-bit"]["mean_gain_db"] - ref for b in bits]
    meas_bound = [quant_bound[f"{b}-bit"]["mean_gain_db"] - ref_b for b in bits]
    theory = [20 * np.log10(np.sinc(1.0 / (2 ** b))) for b in bits]

    w = 0.27
    x = np.arange(len(bits))
    a3.bar(x - w, theory, w, label=r"Theory $20\log_{10}\,\mathrm{sinc}(2^{-b})$",
           color=COLORS["gray"], edgecolor="black", linewidth=0.5)
    a3.bar(x, meas_bound, w, label="Perfect-CSI MRC", color=COLORS["blue"],
           edgecolor="black", linewidth=0.5)
    a3.bar(x + w, meas, w, label=STYLE[scheme]["label"], color=COLORS["red"],
           edgecolor="black", linewidth=0.5)
    a3.set_xticks(x)
    a3.set_xticklabels([f"{b}-bit" for b in bits])
    a3.set_ylabel("Array-gain loss (dB)")
    a3.set_title("(c) Quantization loss", fontsize=9, loc="left")
    a3.grid(True, axis="y", ls="--", lw=0.35, alpha=0.3)
    _style_legend(a3, loc="lower right", fontsize=5.8)

    _save(fig, save_path, "fig4_hardware_impairments")


# ==========================================================================
# Figure 5 — CSI robustness
# ==========================================================================

def plot_csi_robustness(results, save_path):
    blk = results.get("csi_robustness") or {}
    if not blk:
        return
    var = np.array(blk["csi_error_variances"])
    rho = np.array(blk["rho_db"]) + results["meta"]["noise_power_dbm"]
    op = blk["operating_rho_db"] + results["meta"]["noise_power_dbm"]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.16, 3.0))

    ber = blk["ber_at_operating_point"]
    for key in ORDER:
        if key in ber:
            _plot_scheme(a1, var, np.maximum(np.array(ber[key]), BER_FLOOR), key,
                         markevery=1, mask_floor=True)
    a1.set_xscale("symlog", linthresh=1e-3)
    _log_ber_axis(a1)
    a1.set_xlabel(r"CSI error variance $\sigma_e^2$")
    a1.set_ylabel("BER (QPSK)")
    a1.set_title(rf"(a) BER at $P_t$ = {op:g} dBm", fontsize=9, loc="left")

    se = blk["spectral_efficiency_at_operating_point"]
    for key in ORDER:
        if key in se:
            _plot_scheme(a2, var, np.array(se[key]), key, markevery=1)
    a2.set_xscale("symlog", linthresh=1e-3)
    a2.set_xlabel(r"CSI error variance $\sigma_e^2$")
    a2.set_ylabel("Spectral efficiency (bit/s/Hz)")
    a2.set_title("(b) Rate under imperfect CSI", fontsize=9, loc="left")

    # (c) full waterfalls at the worst CSI quality — the regime that separates
    # a learned prior from a point-estimate solver.
    worst = f"{var[-1]:g}"
    curves = blk["ber_curves"][worst]
    for key in ORDER:
        if key in curves:
            _plot_scheme(a3, rho, curves[key], key, mask_floor=True)
    _log_ber_axis(a3)
    a3.set_xlabel(r"Transmit power $P_t$ (dBm)")
    a3.set_title(rf"(c) Waterfall at $\sigma_e^2$ = {worst}", fontsize=9, loc="left")

    _shared_legend(fig, [a2], ncol=4, y=0.005)
    _save(fig, save_path, "fig5_csi_robustness")


# ==========================================================================
# Figure 6 — Array-gain scaling and SNR distribution
# ==========================================================================

def plot_array_scaling(results, save_path):
    blk = results["array_scaling"]
    n = np.array(blk["element_counts"])
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.16, 3.0))

    for key in ORDER:
        if key in blk["reflected_only_snr_db"]:
            _plot_scheme(a1, n, np.array(blk["reflected_only_snr_db"][key]), key, markevery=1)
    anchor = blk["ideal_anchor_index"]
    ideal = np.array(blk["ideal_n_squared_db"])
    a1.plot(n, ideal, color="#444444", ls=(0, (1, 1)), lw=1.0,
            label=rf"$N^2$ law (anchored at $N$={n[anchor]})")
    a1.set_xscale("log", base=2)
    a1.set_xlabel("Active RIS elements $N$")
    a1.set_ylabel("Reflected-only received SNR (dB)")
    a1.set_title("(a) Array-gain scaling", fontsize=9, loc="left")

    for key in ORDER:
        if key in blk["spectral_efficiency"]:
            _plot_scheme(a2, n, np.array(blk["spectral_efficiency"][key]), key, markevery=1)
    a2.set_xscale("log", base=2)
    a2.set_xlabel("Active RIS elements $N$")
    a2.set_ylabel("Spectral efficiency (bit/s/Hz)")
    a2.set_title("(b) Rate vs surface size", fontsize=9, loc="left")

    for key in ORDER:
        if key in blk["snr_cdf"]:
            s = np.array(blk["snr_cdf"][key])
            _plot_scheme(a3, s, np.linspace(0, 1, len(s)), key, markevery=max(1, len(s) // 6))
    a3.set_xlabel("Received SNR (dB)")
    a3.set_ylabel("Empirical CDF")
    a3.set_title("(c) SNR distribution, full surface", fontsize=9, loc="left")

    _shared_legend(fig, [a1], ncol=4, y=0.005)
    _save(fig, save_path, "fig6_array_scaling")


FIGURES = (
    plot_ber_vs_snr,
    plot_spectral_efficiency,
    plot_outage,
    plot_hardware_impairments,
    plot_csi_robustness,
    plot_array_scaling,
)


def render_all(results, save_path):
    os.makedirs(save_path, exist_ok=True)
    for fn in FIGURES:
        fn(results, save_path)
    folder = Path(save_path)
    source = folder / "link_level_results.json"
    meta = results["meta"]
    manifest = {
        "schema": "link-figures-v1", "seed": meta.get("seed"),
        "num_scenes": meta["num_scenes"], "is_quick_run": meta.get("is_quick_run", False),
        "training_status": meta.get("training", {}).get("status", "performed"),
        "source_result": str(source) if source.exists() else None,
        "source_result_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None,
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(folder.glob("fig*")) if p.suffix in (".pdf", ".png")},
        "scope": "per-seed curves; use seed-aggregate report for confidence intervals",
    }
    (folder/"figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

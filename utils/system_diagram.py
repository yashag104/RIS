"""End-to-end system diagram for the federated RIS phase-control architecture.

Draws the whole pipeline in one figure: the 28 GHz propagation scene, the
on-chip federated learning loop that runs across the RIS tiles over the NoC,
and the phase-application and link-level evaluation chain that turns predicted
phases into BER / spectral-efficiency / outage numbers.

    python -m utils.system_diagram [output_dir]

Every number on the figure is read from ``config.Config``, so the diagram cannot
drift away from the configuration the experiments actually run.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from utils.plotting import COLORS, IEEE_RC, _save

plt.rcParams.update(IEEE_RC)

INK = "#222222"
TAN = "#8a5a24"
GRN = "#3f7a44"
BAND = {"phy": "#eaf2f8", "learn": "#fdf0e6", "apply": "#eef7ec"}

# Band extents (axes fraction). Kept in one place so nothing overlaps by accident.
A_Y, A_H = 0.700, 0.272
B_Y, B_H = 0.368, 0.300
C_Y, C_H = 0.030, 0.300


def _box(ax, x, y, w, h, text, fc="white", ec=INK, fs=6.4, lw=0.8,
         weight="normal", zorder=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010",
                                linewidth=lw, facecolor=fc, edgecolor=ec, zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=zorder + 1, weight=weight, linespacing=1.4)
    return x + w


def _arrow(ax, p0, p1, color=INK, lw=0.9, ls="-", rad=0.0, zorder=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5, zorder=zorder))


def _label(ax, x, y, text, fs=5.8, color="#555555", ha="center", va="center",
           style="italic", **kw):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va, style=style,
            zorder=9, linespacing=1.35, **kw)


def _band(ax, y, h, color, title):
    ax.add_patch(Rectangle((0.0, y), 1.0, h, facecolor=color, edgecolor="none", zorder=0))
    ax.text(0.010, y + h - 0.010, title, fontsize=7.4, weight="bold",
            color="#333333", va="top", ha="left", zorder=2)


# --------------------------------------------------------------------------

def _draw_scene(ax, cfg):
    """Band A: BS, blocked direct link, tiled RIS, user."""
    _band(ax, A_Y, A_H, BAND["phy"], "A   Propagation scene")

    base = A_Y + 0.024                      # feet of BS and user
    bs_x, ue_x = 0.085, 0.900

    # Base station.
    ax.add_patch(Rectangle((bs_x - 0.013, base), 0.026, 0.070, facecolor="#c9d9e8",
                           edgecolor=INK, lw=0.8, zorder=3))
    ax.plot([bs_x, bs_x], [base + 0.070, base + 0.104], color=INK, lw=0.9, zorder=4)
    for dy, wdt in ((0.078, 0.030), (0.089, 0.021), (0.099, 0.012)):
        ax.plot([bs_x - wdt / 2, bs_x + wdt / 2], [base + dy, base + dy],
                color=INK, lw=0.9, zorder=4)
    _label(ax, bs_x, base - 0.020,
           f"BS, {cfg.FREQUENCY / 1e9:.0f} GHz\n$P_t$ = {cfg.TX_POWER_DBM} dBm",
           fs=5.6, style="normal")

    # User.
    ax.add_patch(Circle((ue_x, base + 0.086), 0.011, facecolor="#f6d7d5",
                        edgecolor=INK, lw=0.8, zorder=4))
    ax.add_patch(FancyBboxPatch((ue_x - 0.015, base + 0.028), 0.030, 0.048,
                                boxstyle="round,pad=0.004", facecolor="#f6d7d5",
                                edgecolor=INK, lw=0.8, zorder=3))
    _label(ax, ue_x, base - 0.020,
           f"User(s)\n$\\sigma^2$ = {cfg.NOISE_POWER_DBM} dBm", fs=5.6, style="normal")

    # RIS surface.
    rx, ry, rw, rh = 0.420, A_Y + 0.116, 0.158, 0.126
    ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor="#dce9f5", edgecolor=INK,
                           lw=1.1, zorder=3))
    cols, rows = cfg.TILE_GRID_COLS, cfg.TILE_GRID_ROWS
    cw, ch = rw / cols, rh / rows
    for c in range(cols):
        for r in range(rows):
            ax.add_patch(Rectangle((rx + c * cw + 0.0015, ry + r * ch + 0.0015),
                                   cw - 0.003, ch - 0.003, facecolor="white",
                                   edgecolor="#5a7fa6", lw=0.45, zorder=4))
    # One tile highlighted and its element grid drawn in, to show the hierarchy.
    hx, hy = rx + (cols - 1) * cw, ry + (rows - 1) * ch
    ax.add_patch(Rectangle((hx + 0.0015, hy + 0.0015), cw - 0.003, ch - 0.003,
                           facecolor="#fff1dc", edgecolor=COLORS["orange"],
                           lw=1.0, zorder=5))
    pr, pc = cfg.PIXEL_GRID_ROWS, cfg.PIXEL_GRID_COLS
    px, py = (cw - 0.006) / pc, (ch - 0.006) / pr
    for i in range(pc):
        for j in range(pr):
            ax.add_patch(Rectangle((hx + 0.003 + i * px, hy + 0.003 + j * py),
                                   px * 0.68, py * 0.68, facecolor=COLORS["orange"],
                                   edgecolor="none", zorder=6))
    _arrow(ax, (hx + cw, hy + ch / 2), (rx + rw + 0.030, hy + ch / 2),
           color=COLORS["orange"], lw=0.8)
    _label(ax, rx + rw + 0.036, hy + ch / 2,
           f"one tile\n{pr}$\\times${pc} = {cfg.ELEMENTS_PER_TILE} elements",
           fs=5.4, color=COLORS["orange"], ha="left")
    _label(ax, rx - 0.020, ry + rh - 0.018,
           f"RIS surface\n{rows}$\\times${cols} = {cfg.NUM_TILES} tiles\n"
           f"$N$ = {cfg.NUM_TILES * cfg.ELEMENTS_PER_TILE} elements",
           fs=6.0, color="#333333", ha="right", style="normal")

    # Cascaded links.
    _arrow(ax, (bs_x + 0.020, base + 0.098), (rx + 0.014, ry - 0.005),
           color=COLORS["blue"], lw=1.2, rad=-0.18)
    _label(ax, 0.235, ry - 0.030, r"$\mathbf{h}_{\mathrm{BS}\to\mathrm{RIS}}$",
           fs=6.6, color=COLORS["blue"], style="normal")
    _arrow(ax, (rx + rw - 0.014, ry - 0.005), (ue_x - 0.022, base + 0.094),
           color=COLORS["green"], lw=1.2, rad=-0.18)
    _label(ax, 0.775, ry - 0.030, r"$\mathbf{h}_{\mathrm{RIS}\to\mathrm{UE},u}$",
           fs=6.6, color=COLORS["green"], style="normal")

    # Obstructed direct path.
    dy = base + 0.052
    ax.plot([bs_x + 0.016, ue_x - 0.020], [dy, dy], color=COLORS["gray"], lw=1.0,
            ls=(0, (3, 2)), zorder=3)
    ax.add_patch(Rectangle((0.290, dy - 0.028), 0.070, 0.056, facecolor="#d9d9d9",
                           edgecolor=INK, lw=0.7, hatch="////", alpha=0.9, zorder=5))
    _label(ax, 0.325, dy - 0.040, f"blockage  {cfg.DIRECT_LINK_BLOCKAGE_DB:.0f} dB",
           fs=5.6, color="#444444", style="normal")
    _label(ax, 0.640, dy + 0.014, r"$h_{\mathrm{direct}}$ (obstructed)",
           fs=6.2, color="#666666", style="normal")


def _draw_federated(ax, cfg):
    """Band B: per-tile learning and the on-chip aggregation loop."""
    _band(ax, B_Y, B_H, BAND["learn"],
          "B   On-chip federated learning across tiles  (no raw CSI ever leaves a tile)")

    tile_w, tile_h, gap = 0.176, 0.150, 0.024
    top = B_Y + B_H - 0.042
    xs = [0.048 + i * (tile_w + gap) for i in range(3)]
    names = ["RIS tile 1", "RIS tile 2", f"RIS tile {cfg.NUM_TILES}"]
    inner = [
        ("Local CSI est.\n" r"$\hat{\mathbf{h}}$ (pilots)", "white"),
        ("Feature map\nRe / Im", "white"),
        (f"{cfg.MODEL_TYPE} phase net\n" r"$f_{\mathbf{w}}\!:\hat{\mathbf{h}}\mapsto\theta$", "#ffe9c9"),
        ("Local SGD\n" r"$\max$ sum-rate", "white"),
    ]

    bw = (tile_w - 0.032) / 2
    bh = (tile_h - 0.050) / 2
    for k, x in enumerate(xs):
        ax.add_patch(FancyBboxPatch((x, top - tile_h), tile_w, tile_h,
                                    boxstyle="round,pad=0.010", facecolor="#fff6ee",
                                    edgecolor="#b07a44", lw=1.1, zorder=3))
        ax.text(x + tile_w / 2, top - 0.014, names[k], fontsize=6.4, weight="bold",
                ha="center", va="center", color=TAN, zorder=7)
        for j, (txt, fc) in enumerate(inner):
            bx = x + 0.011 + (j % 2) * (bw + 0.010)
            by = top - 0.030 - (j // 2 + 1) * bh - (j // 2) * 0.008
            _box(ax, bx, by, bw, bh, txt, fc=fc, fs=4.6, lw=0.55, zorder=6)
    _label(ax, xs[1] + tile_w + gap / 2, top - tile_h / 2, r"$\cdots$", fs=10,
           color=TAN, style="normal")

    noc_y, noc_h = B_Y + 0.022, 0.056
    _box(ax, 0.048, noc_y, 0.578, noc_h,
         f"Network-on-Chip   |   {cfg.NOC_TOPOLOGY} topology   |   "
         f"{cfg.NOC_PROTOCOL}   |   {cfg.NOC_BANDWIDTH_GBPS} Gb/s\n"
         f"payload: INT8 model deltas ({cfg.COMM_BYTES_PER_PARAM} B/param) "
         "— never channel samples",
         fc="#f3e2cd", ec="#b07a44", lw=1.1, fs=5.7)

    tile_bottom = top - tile_h
    for x in xs:
        cx = x + tile_w / 2
        _arrow(ax, (cx - 0.020, tile_bottom - 0.004), (cx - 0.020, noc_y + noc_h + 0.004),
               color=TAN, lw=0.95)
        _arrow(ax, (cx + 0.020, noc_y + noc_h + 0.004), (cx + 0.020, tile_bottom - 0.004),
               color=TAN, lw=0.95, ls=(0, (2.5, 1.8)))
    mid_y = (tile_bottom + noc_y + noc_h) / 2
    _label(ax, xs[0] + tile_w / 2 - 0.028, mid_y, r"$\Delta\mathbf{w}_k$",
           fs=5.8, color=TAN, ha="right", style="normal")
    _label(ax, xs[0] + tile_w / 2 + 0.028, mid_y, r"$\mathbf{w}^{r+1}$",
           fs=5.8, color=TAN, ha="left", style="normal")

    agg_x, agg_y, agg_w, agg_h = 0.664, noc_y, 0.294, 0.236
    _box(ax, agg_x, agg_y, agg_w, agg_h,
         "Aggregator (on-chip)\n"
         f"{cfg.AGGREGATION_METHOD} / FedProx / SCAFFOLD\n"
         r"$\mathbf{w}^{r+1}=\sum_k \frac{n_k}{n}\,\mathbf{w}_k^{r}$" "\n"
         f"\n{cfg.FL_ROUNDS} rounds $\\times$ {cfg.LOCAL_EPOCHS} local epochs\n"
         "tile sleep scheduling\n+ pixel duty cycling",
         fc="#f3e2cd", ec="#b07a44", lw=1.1, fs=5.9)
    _arrow(ax, (0.628, noc_y + 0.040), (agg_x - 0.002, noc_y + 0.040), color=TAN, lw=1.1)
    _arrow(ax, (agg_x - 0.002, noc_y + 0.014), (0.628, noc_y + 0.014), color=TAN,
           lw=1.1, ls=(0, (2.5, 1.8)))
    _label(ax, 0.645, noc_y + 0.058, "upload", fs=5.0, color=TAN)
    _label(ax, 0.645, noc_y - 0.006, "broadcast", fs=5.0, color=TAN)


def _draw_apply(ax, cfg):
    """Band C: phase application chain and the metrics it feeds."""
    _band(ax, C_Y, C_H, BAND["apply"], "C   Phase application and link-level evaluation")

    row_y, row_h = C_Y + 0.120, 0.110
    chain = [
        (0.030, 0.124, "Per-element\nprediction\n" r"$\theta_{t,n}$"),
        (0.172, 0.124, "Direct-path\noffset\n" r"$+\,\angle h_{\mathrm{direct}}$"),
        (0.314, 0.124, "Phase quantizer\n$b$ = 1 / 2 / 3 bit\nor continuous"),
        (0.456, 0.124, "Hardware jitter\n" r"$\delta\sim\mathcal{N}(0,\sigma_\phi^2)$"),
    ]
    prev = None
    for x, w, txt in chain:
        _box(ax, x, row_y, w, row_h, txt, fc="white", ec=GRN, fs=5.9, lw=0.85)
        if prev is not None:
            _arrow(ax, (prev, row_y + row_h / 2), (x, row_y + row_h / 2), color=GRN, lw=1.0)
        prev = x + w

    comb_x, comb_w = 0.606, 0.176
    _box(ax, comb_x, row_y - 0.014, comb_w, row_h + 0.028,
         "Coherent combining\n"
         r"$h_{\mathrm{eff}}=h_{\mathrm{d}}+\sum_{t,n}c_{t,n}e^{j\theta_{t,n}}$" "\n"
         r"$\gamma=\rho\,|h_{\mathrm{eff}}|^{2}$",
         fc="#e2f0e2", ec=GRN, fs=5.7, lw=1.1)
    _arrow(ax, (prev, row_y + row_h / 2), (comb_x, row_y + row_h / 2), color=GRN, lw=1.0)

    met_x, met_w = 0.806, 0.168
    _box(ax, met_x, row_y - 0.014, met_w, row_h + 0.028,
         "Link-level metrics\nBER / SER vs SNR\nErgodic spectral eff.\n"
         "Outage probability\nArray-gain scaling",
         fc="#e2f0e2", ec=GRN, fs=5.7, lw=1.1, weight="bold")
    _arrow(ax, (comb_x + comb_w, row_y + row_h / 2), (met_x, row_y + row_h / 2),
           color=GRN, lw=1.0)

    _label(ax, 0.42, C_Y + 0.062,
           "designs are built only from the CSI each scheme is entitled to see, "
           "then scored on the TRUE channel",
           fs=5.5, color=GRN)


def build_figure(cfg):
    fig, ax = plt.subplots(figsize=(7.16, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_scene(ax, cfg)
    _draw_federated(ax, cfg)
    _draw_apply(ax, cfg)

    _arrow(ax, (0.300, A_Y - 0.004), (0.300, B_Y + B_H + 0.004), color=INK, lw=1.0)
    _label(ax, 0.310, (A_Y + B_Y + B_H) / 2, "per-tile pilots", fs=5.4, ha="left")
    _arrow(ax, (0.860, B_Y - 0.004), (0.860, C_Y + C_H + 0.004), color=INK, lw=1.0)
    _label(ax, 0.850, (B_Y + C_Y + C_H) / 2, "trained global model", fs=5.4, ha="right")

    # Closing the loop: the applied surface changes the channel the next block sees.
    ax.annotate("", xy=(0.986, A_Y + 0.030), xytext=(0.986, C_Y + 0.090),
                arrowprops=dict(arrowstyle="-|>", color="#8c8c8c", lw=1.0))
    ax.text(0.978, (A_Y + C_Y) / 2 + 0.06, "next coherence block", rotation=90,
            fontsize=5.4, color="#8c8c8c", ha="right", va="center", style="italic")

    fig.suptitle("Federated phase control for a tiled RIS: end-to-end system",
                 fontsize=10.5, weight="bold", y=0.995)
    return fig


def main(out_dir: str = "results/figures") -> str:
    from config import Config

    fig = build_figure(Config)
    _save(fig, out_dir, "system_architecture")
    return os.path.join(out_dir, "system_architecture.pdf")


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else "results/figures"))

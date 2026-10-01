"""Re-render the original Fig. 1 and Fig. 2 on a -5..25 dB received-SNR axis.

Source: results/link_level/link_level_results.json (the Sep-16 seed-42 run that
compares No RIS, Random, AO, SCA, Centralized DL, Fed-RIS and the bound).

That JSON stores its BER/rate curves only from rho = 88 dB (about +7 dB received
SNR), so the low end cannot be read off the stored curves. It also stores every
scheme's per-scene received SNR at rho = 99 dB (array_scaling.snr_cdf, full
1024-element surface, 600 scenes). Received SNR is linear in transmit power, so
the per-scene gains are recovered exactly and the curves are recomputed at any
rho with the same closed-form BER / log2(1+SNR) used by the original run. The
recomputation reproduces every stored curve to machine precision (checked
below). No model is retrained or re-run.

Usage: .venv/bin/python make_link_level_snr_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import utils.plotting_link as pl
from src.link_metrics import ber_curve, spectral_efficiency_curve

SRC = Path("results/link_level/link_level_results.json")
OUT = Path("results/link_level_snr_-5_25")


def main():
    res = json.loads(SRC.read_text())
    op_rho = res["array_scaling"]["operating_rho_db"]
    gains = {k: 10 ** ((np.asarray(s) - op_rho) / 10)
             for k, s in res["array_scaling"]["snr_cdf"].items()}

    # Exactness check against the stored curves.
    old_rho = np.asarray(res["ber_vs_snr"]["rho_db"])
    for k, g in gains.items():
        for mod, stored in res["ber_vs_snr"]["modulations"].items():
            assert np.allclose(ber_curve(g, old_rho, mod), stored[k], rtol=1e-9), (k, mod)
        assert np.allclose(spectral_efficiency_curve(g, old_rho),
                           res["spectral_efficiency"]["spectral_efficiency"][k], rtol=1e-9), k

    # Reference received SNR = rho + genie mean gain; cover -5 dB up to the
    # original top of the sweep (panel (b) of Fig. 2 needs the no-RIS curve to
    # reach 4 bit/s/Hz, which happens only at high power).
    ref = res["mean_gain_db"]["genie"]
    rho = np.arange(-5.0 - ref, old_rho[-1] + 0.01, 0.25)
    res["ber_vs_snr"] = {
        "rho_db": rho.tolist(),
        "modulations": {m: {k: ber_curve(g, rho, m).tolist() for k, g in gains.items()}
                        for m in ("QPSK", "16QAM")},
    }
    res["spectral_efficiency"] = {
        "rho_db": rho.tolist(),
        "spectral_efficiency": {k: spectral_efficiency_curve(g, rho).tolist()
                                for k, g in gains.items()},
    }

    # Keep the scheme names the original figures used.
    pl.STYLE["ao"]["label"] = "Alternating Opt."
    pl.STYLE["sca"]["label"] = "SCA"
    pl.SNR_AXIS_LIMITS = (-5.0, 25.0)

    # Linear-y panels (the rate plot) must scale to what is visible in
    # -5..25 dB, not to the high-power tail kept only for Fig. 2(b).
    base_axis = pl._apply_snr_axis

    def snr_axis(ax):
        base_axis(ax)
        if ax.get_yscale() == "linear":
            lo, hi = pl.SNR_AXIS_LIMITS
            ys = [y for ln in ax.get_lines() for x, y in zip(ln.get_xdata(), ln.get_ydata())
                  if lo <= x <= hi and np.isfinite(y)]
            ax.set_ylim(min(0.0, min(ys)) - 0.2, max(ys) * 1.05)

    pl._apply_snr_axis = snr_axis

    pl.plot_ber_vs_snr(res, OUT)
    pl.plot_spectral_efficiency(res, OUT)
    (OUT / "curves.json").write_text(json.dumps({
        "source": str(SRC), "reference_gain_db": ref,
        "reference_snr_db": (rho + ref).tolist(),
        "ber_vs_snr": res["ber_vs_snr"], "spectral_efficiency": res["spectral_efficiency"],
    }))
    print(f"wrote {OUT}/fig1_ber_vs_snr.(png|pdf), fig2_spectral_efficiency.(png|pdf), curves.json")


if __name__ == "__main__":
    main()

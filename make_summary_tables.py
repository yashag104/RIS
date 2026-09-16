#!/usr/bin/env python
"""Emit the Markdown tables used in docs/ from the saved result JSONs.

Numbers in a paper should be traceable to a file, not retyped. Running

    python make_summary_tables.py

prints every table in docs/BASELINE_COMPARISON.md and docs/PAPER_RESULTS.md so
they can be regenerated after any re-run instead of drifting.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

LINK_JSON = "results/link_level/link_level_results.json"
ADV_DIR = "results/advanced_experiments"

SCHEME_ORDER = ["no_ris", "random_ris", "ao", "sca", "centralized_dl", "fed_ris", "genie"]
PRETTY = {
    "no_ris": "No RIS (blocked direct)",
    "random_ris": "Random phases",
    "ao": "Alternating Optimization",
    "sca": "SCA",
    "centralized_dl": "Centralized DL (pooled)",
    "fed_ris": "**Fed-RIS (proposed)**",
    "genie": "Perfect-CSI MRC (bound)",
}
# What each scheme costs to run, independent of the numbers measured here.
PROPERTIES = {
    "no_ris":         ("—", "—", "yes", "O(1)"),
    "random_ris":     ("none", "none", "yes", "O(N)"),
    "ao":             ("full instantaneous", "per block", "no", "O(N·I)"),
    "sca":            ("full instantaneous", "per block", "no", "O(N·I)"),
    "centralized_dl": ("pooled to server", "one forward", "no", "O(N)"),
    "fed_ris":        ("stays on tile", "one forward", "yes", "O(N)"),
    "genie":          ("perfect (oracle)", "closed form", "n/a", "O(N)"),
}


def _fmt(v, nd=2, dash="—"):
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return dash if not np.isfinite(f) else f"{f:.{nd}f}"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def table_link_summary(res):
    """Main scheme comparison: array gain, SNR at target BER, rate."""
    gains = res["mean_gain_db"]
    tgt = res["ber_vs_snr"]["snr_at_target"]
    se = res["spectral_efficiency"]["spectral_efficiency"]
    rho = np.array(res["spectral_efficiency"]["rho_db"])
    op = res["meta"].get("operating_rho_db")
    ref = gains["no_ris"]
    fed_q = tgt["QPSK"]["fed_ris"]["1e-03"]

    rows = []
    for k in SCHEME_ORDER:
        se_at_op = float(np.interp(op, rho, se[k])) if op else float("nan")
        rows.append([
            PRETTY[k],
            _fmt(gains[k] - ref),
            _fmt(tgt["QPSK"][k]["1e-03"], 1),
            _fmt(tgt["QPSK"][k]["1e-04"], 1),
            _fmt(tgt["16QAM"][k]["1e-03"], 1),
            _fmt(se_at_op),
            _fmt(tgt["QPSK"][k]["1e-03"] - fed_q, 1),
        ])
    return _table(
        ["Scheme", "Array gain vs no-RIS (dB)", "ρ @ BER 1e-3, QPSK (dB)",
         "ρ @ BER 1e-4, QPSK (dB)", "ρ @ BER 1e-3, 16QAM (dB)",
         f"SE @ ρ={_fmt(op, 0)} dB (bit/s/Hz)", "Δ vs Fed-RIS (dB)"],
        rows,
    )


def table_properties():
    rows = [[PRETTY[k], *PROPERTIES[k]] for k in SCHEME_ORDER]
    return _table(["Scheme", "CSI requirement", "Per-block cost", "CSI stays local",
                   "Inference complexity"], rows)


def table_quantization(res):
    q = res["hardware_impairments"]["quantization"]
    rows = []
    for bits in (1, 2, 3):
        label = f"{bits}-bit"
        theory = 20 * np.log10(np.sinc(1.0 / (2 ** bits)))
        rows.append([
            f"{label} ({2 ** bits} states)",
            _fmt(theory),
            _fmt(q["genie"][label]["mean_gain_db"] - q["genie"]["continuous"]["mean_gain_db"]),
            _fmt(q["fed_ris"][label]["mean_gain_db"] - q["fed_ris"]["continuous"]["mean_gain_db"]),
        ])
    rows.append(["Continuous", "0.00", "0.00", "0.00"])
    return _table(["Phase resolution", "Theory 20·log10 sinc(2⁻ᵇ) (dB)",
                   "Measured, perfect-CSI MRC (dB)", "Measured, Fed-RIS (dB)"], rows)


def table_phase_noise(res):
    pn = res["hardware_impairments"]["phase_noise"]["fed_ris"]
    ref = pn["0deg"]["mean_gain_db"]
    rows = [[k.replace("deg", "°"), _fmt(v["mean_gain_db"] - ref)]
            for k, v in sorted(pn.items(), key=lambda kv: float(kv[0].replace("deg", "")))]
    return _table(["RMS phase jitter σ_φ", "Array-gain loss (dB)"], rows)


def table_csi_robustness(res):
    blk = res.get("csi_robustness") or {}
    if not blk:
        return "_(CSI robustness sweep not present in this run.)_"
    var = blk["csi_error_variances"]
    snr = blk["mean_snr_db"]
    rows = []
    for k in SCHEME_ORDER:
        if k not in snr:
            continue
        vals = snr[k]
        rows.append([PRETTY[k]] + [_fmt(v, 1) for v in vals] + [_fmt(vals[0] - vals[-1], 1)])
    return _table(["Scheme"] + [f"ε={v:g}" for v in var] + ["Loss over sweep (dB)"], rows)


def table_array_scaling(res):
    blk = res["array_scaling"]
    n = blk["element_counts"]
    rows = []
    for k in SCHEME_ORDER:
        rows.append([PRETTY[k]] + [_fmt(v, 1) for v in blk["mean_snr_db"][k]])
    rows.append(["_Ideal N² law_"] + [_fmt(v, 1) for v in blk["ideal_n_squared_db"]])
    return _table(["Scheme"] + [f"N={v}" for v in n], rows)


def table_cost(res):
    """Training and inference cost, including the traffic federation actually costs.

    The FL-vs-raw-CSI comparison is reported both ways round on purpose. With a
    300k-parameter model and a few hundred samples per tile, shipping weights
    every round moves far *more* bytes than shipping the raw CSI once — the
    federated design buys locality here, not bandwidth. The crossover row says
    where the trade flips.
    """
    tr = res["meta"].get("training", {})
    params = tr.get("model_parameters") or 0
    rounds = tr.get("fl_rounds") or 0
    samples = tr.get("train_samples_per_tile") or 0
    fl_kb = tr.get("fl_total_communication_kb")
    csi_kb = tr.get("raw_csi_upload_kb")

    # FL moves 2 * rounds * params bytes per tile (upload + broadcast, INT8);
    # a centralized controller moves 4 * samples * feature_dim bytes per tile
    # once. Equate them to get the per-tile dataset size at which FL is cheaper.
    feature_dim = (samples and csi_kb and (csi_kb * 1024) /
                   (res["meta"]["num_tiles"] * samples * 4)) or 0
    crossover = (2 * rounds * params / (4 * feature_dim)) if feature_dim else float("nan")

    rows = [
        ["Model parameters", f"{params:,}"],
        ["FL rounds × local epochs", f"{rounds} × {tr.get('local_epochs')}"],
        ["Tiles", f"{res['meta']['num_tiles']}"],
        ["Train samples per tile", f"{samples:,}"],
        ["Total NoC traffic, FL, INT8 deltas (MB)",
         _fmt((fl_kb or 0) / 1024, 1)],
        ["Raw-CSI upload a centralized controller needs (MB)",
         _fmt((csi_kb or 0) / 1024, 1)],
        ["FL traffic ÷ raw-CSI traffic",
         _fmt((fl_kb / csi_kb) if (fl_kb and csi_kb) else None, 1) + "×"],
        ["Samples per tile at which FL becomes the cheaper transfer",
         f"≈ {crossover:,.0f}" if np.isfinite(crossover) else "—"],
        ["FL wall clock (s)", _fmt(tr.get("fl_wall_clock_s"), 0)],
        ["Centralized wall clock (s)", _fmt(tr.get("centralized_wall_clock_s"), 0)],
    ]
    st = res["meta"].get("mean_solve_time_s", {})
    for k, v in st.items():
        rows.append([f"Mean per-coherence-block solve time, {PRETTY[k]} (ms)",
                     _fmt(v * 1e3, 3)])
    rows.append(["Mean per-coherence-block inference, learned schemes",
                 "one forward pass, no iteration"])
    return _table(["Quantity", "Value"], rows)


def table_legacy_baselines():
    """The system-level experiment-9 table, for continuity with the earlier runs."""
    path = os.path.join(ADV_DIR, "baseline_comparison_results.json")
    if not os.path.exists(path):
        return "_(results/advanced_experiments/baseline_comparison_results.json not found.)_"
    with open(path) as fh:
        blob = json.load(fh)
    entry = blob["results"][0] if isinstance(blob["results"], list) else blob["results"]
    prov = blob.get("provenance", {})
    rows = []
    for key, val in entry.items():
        if not isinstance(val, dict):
            continue
        rows.append([
            key, _fmt(val.get("snr_db")), _fmt(val.get("rate_bps_hz")),
            _fmt(val.get("communication_kb"), 1), _fmt(val.get("energy_mj"), 1),
            str(val.get("convergence_iters", "—")),
            "yes" if val.get("privacy") else "no",
            str(val.get("complexity", "—")),
        ])
    note = (f"\n\n_Provenance: {prov.get('num_tiles')} tiles × "
            f"{prov.get('elements_per_tile')} elements, {prov.get('fl_rounds')} FL rounds, "
            f"{prov.get('train_samples')} train samples, "
            f"reduced run = {prov.get('is_reduced_run')}, saved {prov.get('saved_at')}._")
    return _table(["Method", "SNR (dB)", "Rate (bit/s/Hz)", "Comm (KB)", "Energy (mJ)",
                   "Iterations", "CSI stays local", "Complexity"], rows) + note


BEGIN = "<!-- BEGIN GENERATED TABLES -->"
END = "<!-- END GENERATED TABLES -->"


def build_sections(res):
    meta = res["meta"]
    provenance = "## Run provenance\n\n" + _table(["Setting", "Value"], [
        ["Tiles × elements", f"{meta['num_tiles']} × {meta['elements_per_tile']} "
                             f"= {meta['total_elements']}"],
        ["Test scenes", meta["num_scenes"]],
        ["Carrier", f"{meta['frequency_hz'] / 1e9:.0f} GHz"],
        ["Noise power", f"{meta['noise_power_dbm']} dBm"],
        ["Direct-link blockage", f"{meta['direct_link_blockage_db']} dB"],
        ["Operating point ρ", f"{_fmt(meta.get('operating_rho_db'), 0)} dB"],
        ["Reduced (quick) run", meta.get("is_quick_run")],
        ["Seed", meta.get("seed")],
    ])
    return [
        ("provenance", provenance),
        ("Table 1 — Link-level comparison, all schemes on one channel set",
         table_link_summary(res)),
        ("Table 2 — What each scheme requires to produce a phase design",
         table_properties()),
        ("Table 3 — Phase-quantization loss vs the classical sinc bound",
         table_quantization(res)),
        ("Table 4 — RIS phase-jitter loss (Fed-RIS design)", table_phase_noise(res)),
        ("Table 5 — Mean received SNR vs normalized CSI error ε",
         table_csi_robustness(res)),
        ("Table 6 — Array-gain scaling (mean received SNR, dB)",
         table_array_scaling(res)),
        ("Table 7 — Training and per-coherence-block cost", table_cost(res)),
        ("Table 8 — System-level baseline table (experiment 9, earlier run)",
         table_legacy_baselines()),
    ]


def render(res, only=None):
    parts = []
    for title, body in build_sections(res):
        if only and title.split(" —")[0] not in only and title not in only:
            continue
        parts.append(body if title == "provenance" else f"## {title}\n\n{body}")
    return "\n\n".join(parts)


def splice(path: str, text: str) -> bool:
    """Replace the generated block in a Markdown file, leaving prose untouched."""
    with open(path) as fh:
        doc = fh.read()
    if BEGIN not in doc or END not in doc:
        print(f"  ! {path} has no generated-table markers; skipped")
        return False
    new = re.sub(
        re.escape(BEGIN) + r".*?" + re.escape(END),
        f"{BEGIN}\n<!-- regenerate with: python make_summary_tables.py --write -->\n\n"
        f"{text}\n\n{END}",
        doc,
        flags=re.DOTALL,
    )
    if new == doc:
        return False
    with open(path, "w") as fh:
        fh.write(new)
    print(f"  > updated {path}")
    return True


#: Which tables belong in which document.
DOC_TABLES = {
    "docs/BASELINE_COMPARISON.md": None,          # all of them
    "docs/PAPER_RESULTS.md": ["provenance", "Table 1", "Table 3", "Table 5", "Table 6"],
    "docs/NOVELTY.md": ["provenance", "Table 1", "Table 2", "Table 7"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="splice the tables into the docs instead of printing them")
    args = ap.parse_args()

    if not os.path.exists(LINK_JSON):
        raise SystemExit(f"missing {LINK_JSON}; run `python run_link_level.py` first")
    with open(LINK_JSON) as fh:
        res = json.load(fh)

    if not args.write:
        print(render(res))
        return
    for path, only in DOC_TABLES.items():
        if os.path.exists(path):
            splice(path, render(res, only))
        else:
            print(f"  ! {path} not found; skipped")


if __name__ == "__main__":
    main()

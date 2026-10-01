#!/usr/bin/env python
"""Report the requested M=1025 diagnostic on power metrics, not net rate."""
import hashlib
import json
from pathlib import Path

from make_summary_tables import LABELS
from run_link_level import seed_statistics, write_json


def main():
    out = Path("results/pilot_positive_control")
    paths = [out/f"seed_{seed}/M1025_Pt30/results.json" for seed in (42, 123, 456)]
    rows = [json.loads(p.read_text()) for p in paths]
    for r in rows:
        assert r["meta"]["num_probes"] == 1025
        assert r["meta"]["sample_counts_train_validation_test"] == [600, 200, 600]
        assert r["training"]["max_rounds"] == 100
        assert r["training"]["model_type"] == "MLP"
    metrics = ("fraction_of_oracle_power", "snr_gap_to_local_linear_mrc_db")
    stats = {k: {m: seed_statistics([r["scores"][k][m] for r in rows]) for m in metrics}
             for k in rows[0]["scores"]}
    def cell(stat):
        return f"{stat['mean']:.4f} ± {stat['ci95_half_width']:.4f}"
    md = ["# Full-probe-count pilot diagnostic", "",
          "Exact requested full-baseline runner: M=1025, seeds 42/123/456, 100-round cap, one CPU thread.",
          "600/200/600 training/validation/test scenes per seed. All controllers retain their original inputs and training budgets.",
          "Metrics below exclude the pilot-overhead prelog. Intervals are 95% Student-t intervals across three seeds; paired SNR gaps use the same seed.", "",
          "| Method | Fraction of oracle power | SNR gap to local linear MRC (dB) |", "|---|---:|---:|"]
    tex = [r"\begin{table*}[t]\centering\small",
           r"\caption{Full-probe-count diagnostic at $M=1025$, $P_t=30$\,dBm, three seeds, and a 100-round cap. Metrics exclude the pilot-overhead prelog. Intervals are 95\% seed-level Student-$t$ intervals.}",
           r"\label{tab:positive-control}\begin{tabular}{lrr}\toprule",
           r"Method & Fraction of oracle power & Paired SNR gap to local linear MRC (dB)\\\midrule"]
    for k, v in stats.items():
        a, b = (cell(v[m]) for m in metrics)
        md.append(f"| {LABELS[k]} | {a} | {b} |")
        tex.append(f"{LABELS[k]} & {a.replace('±', r'$\pm$')} & {b.replace('±', r'$\pm$')} "+r"\\")
    tex += [r"\bottomrule\end{tabular}\end{table*}"]
    stopping = [{"seed": r["meta"]["seed"], "rounds": r["training"]["fl_rounds"],
                 "selected_round": r["training"]["best_round"],
                 "reason": r["training"]["stopping_reason"]} for r in rows]
    md += ["", "Training stopping records:", "", *[f"- {r}" for r in stopping], "",
           "M=N+1 removes dimensional underdetermination only if the probe matrix is full rank. The random codebook is not an orthogonal DFT acquisition, and finite-SNR CSI is not perfect CSI. The full-probe LS control uses an orthogonal DFT codebook. Neither this experiment nor a poor neural result alone proves or excludes a shared optimizer/surrogate bug."]
    check_path = out/"observability_check.json"
    check = json.loads(check_path.read_text())
    md += ["", f"The fixed random probe matrix has rank {check['rank']} and condition number {check['condition_number']:.2f}.",
           f"The separate noiseless feature/inverse/phase check reaches oracle-power fraction {check['scores']['float32_feature_inverse']['fraction_of_oracle_power']:.10f} after the same float32 input normalization.",
           "That confirms this algebraic input/phase path preserves the optimum in the checked scenes; it does not validate neural training or the local surrogate under uncertainty."]
    inputs = paths+[check_path]
    manifest = {"inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
                "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "readout": list(metrics), "net_rate_used_for_diagnosis": False}
    write_json(out/"power_summary.json", {"seeds": [42, 123, 456], "scores": stats, "stopping": stopping})
    write_json(out/"report_manifest.json", manifest)
    (out/"REPORT.md").write_text("\n".join(md)+"\n")
    (out/"tables.tex").write_text("\n".join(tex)+"\n")
    print(out/"REPORT.md")


if __name__ == "__main__":
    main()

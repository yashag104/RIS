#!/usr/bin/env python
"""Export Tier 2 tables and figures from completed, provenance-carrying JSON."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_link_level import seed_statistics, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--architectures", default="results/pilot_architecture")
    ap.add_argument("--noc", default="results/noc_corrected/results.json")
    ap.add_argument("--output", default="results/tier2_report")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"inputs": [], "legacy_log_used": False, "supplied_csi_gat_retrained": False}

    def read(path):
        path = Path(path)
        manifest["inputs"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return json.loads(path.read_text())

    arch = read(Path(args.architectures)/"summary.json")
    execution = read(Path(args.architectures)/"execution_manifest.json")
    runner = execution["runner"]
    assert hashlib.sha256(Path(runner["path"]).read_bytes()).hexdigest() == runner["sha256"]
    manifest["architecture_execution_runner"] = runner
    assert arch["schema"] == "pilot-architecture-v1"
    assert len(arch["seeds"]) >= 5 and len(set(arch["seeds"])) == len(arch["seeds"])
    cases = [read(Path(args.architectures)/f"seed_{seed}.json") for seed in arch["seeds"]]
    for r in cases:
        assert r["meta"]["sample_counts_train_validation_test"] == [600, 200, 600]
        assert not r["meta"]["supplied_csi_gat_retraining"]
        assert r["meta"]["arguments"]["probes"] == 16
        assert r["meta"]["arguments"]["tx_power_dbm"] == 30
        assert r["meta"]["source_sha256"]
    steps = {v["optimizer_steps"] for r in cases for v in r["architectures"].values()}
    assert len(steps) == 1
    budget = steps.pop()
    labels = {"MLP": "MLP", "GNN": "GAT (compact)", "CNN": "CNN + SE", "Transformer": "Transformer"}
    metrics = ("mean_received_snr_db", "net_spectral_efficiency")
    # Recompute every exported statistic from per-seed files; summary is audited.
    for key in labels:
        for metric in metrics:
            stat = seed_statistics([r["architectures"][key]["scores"][metric] for r in cases])
            assert np.isclose(stat["mean"], arch["architectures"][key]["scores"][metric]["mean"])
            arch["architectures"][key]["scores"][metric] = stat
            arch["architectures"][key]["paired_"+metric+"_vs_mlp"] = seed_statistics([
                r["architectures"][key]["scores"][metric]-r["architectures"]["MLP"]["scores"][metric]
                for r in cases])
    def fmt(s):
        return f"{s['mean']:.3f} $\\pm$ {s['ci95_half_width']:.3f}"
    def mdcell(s):
        return fmt(s).replace("$", "").replace("\\pm", "±")
    md = ["# Tier 2 corrected results", "",
          f"Architecture diagnostic: five seeds, {budget} pooled Adam updates each; M=16, Pt=30 dBm, L=2048.",
          "600/200/600 train/validation/test scenes; shared pilot acquisition and scoring harness.",
          "Validation-selected checkpoints. Budget-limited, unequal FLOPs/parameter counts; no convergence or supplied-CSI GAT claim.", "",
          "| Pilot model | Parameters | Received SNR (dB), 95% CI | Net SE (bit/s/Hz), 95% CI | Paired net-SE gap vs MLP |",
          "|---|---:|---:|---:|---:|"]
    tex = [r"\begin{table*}[t]\centering\small",
           rf"\caption{{Pilot-feedback architecture diagnostic: five seeds, {budget} pooled updates per model, $M=16$, $P_t=30$\,dBm. Seed-level 95\% intervals; compact profiles, unequal FLOPs. Checkpoints are selected using validation. This is not supplied-CSI GAT retraining.}}",
           r"\label{tab:pilot-architecture}\begin{tabular}{lrrrr}\toprule",
           r"Model & Parameters & Received SNR (dB) & Net rate (bit/s/Hz) & Paired net-rate gap to MLP\\\midrule"]
    for k, label in labels.items():
        v = arch["architectures"][k]
        snr, rate, gap = v["scores"][metrics[0]], v["scores"][metrics[1]], v["paired_"+metrics[1]+"_vs_mlp"]
        tex.append(f"{label} & {v['parameters']} & {fmt(snr)} & {fmt(rate)} & {fmt(gap)} "+r"\\")
        md.append(f"| {label} | {v['parameters']} | {mdcell(snr)} | {mdcell(rate)} | {mdcell(gap)} |")
    tex += [r"\bottomrule\end{tabular}\end{table*}"]
    md += ["", "Intervals are marginal/exploratory, not simultaneous multiple-comparison intervals. No universal architecture ranking is inferred."]
    findings = ["At the fixed 500-update pilot budget, paired net-rate differences to the MLP are "]
    findings.append(", ".join(labels[k]+" "+fmt(arch["architectures"][k]["paired_net_spectral_efficiency_vs_mlp"])
                              for k in ("GNN", "CNN", "Transformer"))+r"\,bit/s/Hz.")
    findings.append("These exploratory intervals do not establish a universal architecture ranking or convergence of any model. The historical GAT superiority claim is not reinstated.")
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.bar(list(labels.values()), [arch["architectures"][k]["scores"][metrics[1]]["mean"] for k in labels],
           yerr=[arch["architectures"][k]["scores"][metrics[1]]["ci95_half_width"] for k in labels],
           color="#4477AA", capsize=4)
    ax.set_ylabel("Pilot-adjusted rate (bit/s/Hz)")
    ax.set_title(f"M=16, Pt=30 dBm; {budget} pooled updates; five seeds")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out/f"pilot_architectures.{ext}", dpi=180)
    plt.close(fig)
    noc = read(args.noc)
    assert noc["schema"] == "analytical-noc-v2"
    lookup = {(r["topology"], r["protocol"], r["clock_ghz"]): r for r in noc["results"]
              if r["scenario"] == "fixed20_150528bytes"}
    for r in lookup.values():
        assert r["total_energy_nj"] is None and not r["model_assumptions"]["cycle_accurate"]
    topologies = ("Mesh", "Torus", "FoldedTorus", "Tree", "Butterfly", "Hypercube", "Ring")
    md += ["", "Analytical interconnect sensitivity: 16 tiles, 150,528-byte FP32 model, 20 rounds.",
           "128-bit links and endpoint ports at assumed 1/2 GHz (128/256 Gb/s); server at tile 0.",
           "No cycle accuracy, Noxim validation, physical layout, energy, area, or technology-node result.", "",
           "| Logical topology | Endpoint diameter | PS at 128 Gb/s (ms) | RingAR at 128 Gb/s (ms) | RingAR at 256 Gb/s (ms) |",
           "|---|---:|---:|---:|---:|"]
    tex += [r"\begin{table*}[t]\centering\small",
            r"\caption{Analytical interconnect sensitivity for 16 endpoints, a 150,528-byte FP32 model, and 20 rounds. PS is ParameterServer; RingAR is RingAllReduce. Equal link rates do not imply equal silicon area. Energy is uncalibrated.}",
            r"\label{tab:noc}\begin{tabular}{lrrrr}\toprule",
            r"Topology & Endpoint diameter & PS, 128 Gb/s (ms) & RingAR, 128 Gb/s (ms) & RingAR, 256 Gb/s (ms)\\\midrule"]
    for t in topologies:
        a, b, c = [lookup[(t, p, f)] for p, f in (("ParameterServer", 1.0), ("RingAllReduce", 1.0), ("RingAllReduce", 2.0))]
        vals = f"{a['topology_diameter']} & {a['total_latency_ms']:.6f} & {b['total_latency_ms']:.6f} & {c['total_latency_ms']:.6f}"
        tex.append(t+" & "+vals+r"\\")
        md.append("| "+t+" | "+vals.replace(" & ", " | ")+" |")
    tex += [r"\bottomrule\end{tabular}\end{table*}"]
    md += ["", "PS serialization alone is 5.644800 ms for every topology at 128 Gb/s; traversal adds 0.000480–0.000960 ms.",
           "A smaller diameter does not remove the endpoint payload. Ring placement/routing changes physical link contention.",
           "Torus and folded torus coincide without a physical wire-layout model. Butterfly has 80 auxiliary stage vertices; it is not the former mislabeled 16-node hypercube.",
           "The complete JSON also replays every retained pilot run's measured model size and FL round count, and includes recursive-doubling and fixed-budget gossip. Gossip is not exact global averaging.", "",
           "Tier 1 #8 remains open for the original supplied-CSI GAT. The retained pilot MLP runs have operational validation plateaus only."]
    write_json(out/"architecture_summary.json", arch)
    (out/"tables.tex").write_text("\n".join(tex)+"\n")
    (out/"findings.tex").write_text("\n".join(findings)+"\n")
    (out/"REPORT.md").write_text("\n".join(md)+"\n")
    manifest["exporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for script in ("run_pilot_ablation.py", "run_noc_study.py"):
        manifest[script+"_sha256"] = hashlib.sha256(Path(script).read_bytes()).hexdigest()
    write_json(out/"manifest.json", manifest)
    print(out/"REPORT.md")


if __name__ == "__main__":
    main()

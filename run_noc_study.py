#!/usr/bin/env python
"""Reproducible analytical interconnect sensitivity, not circuit validation."""
import argparse
import hashlib
import json
from pathlib import Path

from src.noc_simulator import NoCSimulator

TOPOLOGIES = ("Mesh", "Torus", "FoldedTorus", "Tree", "Butterfly", "Hypercube", "Ring")
PROTOCOLS = ("ParameterServer", "AllReduce", "RingAllReduce", "Gossip")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="results/noc_corrected")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Measured parameter counts from the completed pilot training artifacts.
    paths = sorted(Path("results/pilot_limited").glob("seed_*/M*_Pt*/results.json"))
    cases, files = [], []
    for path in paths:
        extended = Path("results/pilot_limited_extended") / path.relative_to("results/pilot_limited")
        if extended.exists():
            path = extended
        r = json.loads(path.read_text())
        tr = r["training"]
        # Field names are explicitly checked; do not invent model sizes.
        size = tr["model_parameters"]*4
        cases.append({"name": f"M{r['meta']['num_probes']}_seed{r['meta']['seed']}",
                          "model_size_bytes": size, "num_rounds": tr["fl_rounds"]})
        files.append(path)
    sizes = {c["model_size_bytes"] for c in cases}
    scenarios = [{"name": f"fixed20_{size}bytes", "model_size_bytes": size, "num_rounds": 20}
                 for size in sorted(sizes)] + cases
    rows = []
    for clock in (1.0, 2.0):
        for topology in TOPOLOGIES:
            sim = NoCSimulator(16, topology, link_width_bits=128, clock_ghz=clock)
            for scenario in scenarios:
                for protocol in PROTOCOLS:
                    rows.append(dict(scenario=scenario["name"], clock_ghz=clock,
                        topology_info=sim.get_topology_info(), **sim.simulate_full_fl_training(
                            scenario["model_size_bytes"], scenario["num_rounds"], protocol)))
    sources = [Path(__file__), Path("src/noc_simulator.py")] + files
    manifest = {str(p.relative_to(Path.cwd()) if p.is_absolute() else p):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result = {"schema": "analytical-noc-v2", "source_sha256": manifest, "scenarios": scenarios,
        "scope": "controller network sensitivity; no RF-aperture-to-die mapping, area, power, or Noxim validation",
        "statistics": "deterministic estimates; no statistical confidence intervals",
        "payload_scope": "FP32 model parameters; server on tile 0, so PS payload is 2*(T-1)*R*model_bytes; excludes data delivery",
        "links": "128 bits/cycle at assumed 1 and 2 GHz; full duplex, same rate at each endpoint",
        "energy_status": "not reported: no technology-calibrated energy model", "results": rows}
    (out/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Saved {len(rows)} analytical cases to {out}")


if __name__ == "__main__":
    main()

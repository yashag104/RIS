#!/usr/bin/env python
"""Noiseless full-probe round-trip diagnostic, separate from learned results.

Checks whether feature normalization and phase application preserve the known
SISO optimum. This tests algebraic identifiability, not finite-SNR acquisition,
sample efficiency, neural optimization, or the local surrogate's statistical fit.
"""
import hashlib
from pathlib import Path

import numpy as np

from run_link_level import write_json
from run_pilot_limited import score
from src.pilot_probing import (
    mrc_from_estimate,
    noiseless_observations,
    pilot_features,
    probe_codebook,
    probe_matrix,
    stack_channel,
)
from src.surface_channel import SceneConfig, SurfaceGeometry, generate_surface_channels


def main():
    geometry, scene = SurfaceGeometry(), SceneConfig()
    n, rho = geometry.total_elements, 1e12
    phases = probe_codebook(n+1, n, 2024)
    A = probe_matrix(phases)
    singular_values = np.linalg.svd(A, compute_uv=False)
    channels = generate_surface_channels(8, geometry, scene, np.random.default_rng(42))
    x = stack_channel(channels["h_direct"], channels["cascade"])
    y = noiseless_observations(channels["h_direct"], channels["cascade"], phases)
    recovered = np.linalg.solve(A, y.T).T
    features = pilot_features(y, rho, geometry.tile_grid_coords())
    # Inverting normalized responses recovers x up to the common rotation and
    # positive scale, both of which cancel in relative MRC reflection phases.
    normalized = features[0, :, :n+1] + 1j*features[0, :, n+1:2*(n+1)]
    recovered_normalized = np.linalg.solve(A, normalized.T).T
    designs = {"oracle": x, "float64_probe_inverse": recovered,
               "float32_feature_inverse": recovered_normalized}
    metrics = {k: score(channels, mrc_from_estimate(v, geometry.num_tiles), rho,
                         n+1, 2048) for k, v in designs.items()}
    result = {"scope": "noiseless algebraic round-trip only; not a learned or deployable result",
              "seed": 42, "codebook_seed": 2024, "samples": 8, "M": n+1, "N": n,
              "rank": int(np.sum(singular_values > singular_values[0]*(n+1)*np.finfo(float).eps)),
              "condition_number": float(singular_values[0]/singular_values[-1]),
              "singular_value_min": float(singular_values[-1]),
              "singular_value_max": float(singular_values[0]),
              "relative_channel_reconstruction_error": float(np.linalg.norm(recovered-x)/np.linalg.norm(x)),
              "scores": metrics,
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                  Path(__file__), Path("src/pilot_probing.py"), Path("src/surface_channel.py"),
                  Path("run_pilot_limited.py"))}}
    assert result["rank"] == n+1
    assert result["relative_channel_reconstruction_error"] < 1e-8
    assert metrics["float32_feature_inverse"]["fraction_of_oracle_power"] > .99999
    write_json("results/pilot_positive_control/observability_check.json", result)
    print("Full-rank and noiseless feature/phase round-trip checks passed.", flush=True)


if __name__ == "__main__":
    main()

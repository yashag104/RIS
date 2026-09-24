"""Analytical hand checks for the revised interconnect and study contracts."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.noc_simulator import NoCSimulator


def test_partial_flit_and_endpoint_serialization_hand_calculation():
    sim = NoCSimulator(4, "Mesh")
    # Three 17-byte messages consume two 16-byte flits each at server ejection.
    p = sim._phase_cost([(1, 0, 17), (2, 0, 17), (3, 0, 17)])
    assert p["payload_bytes"] == 51
    assert p["busy_ns"] == 3*32*8/128
    assert "ejection:0" in p["bottleneck_resources"]
    assert p["traversal_ns"] == 2*3
    assert p["latency_ns"] == 12


def test_full_duplex_endpoint_limits_can_be_slower_than_links():
    sim = NoCSimulator(4, "Mesh", endpoint_bandwidth_gbps=64)
    p = sim._phase_cost([(1, 0, 16), (2, 0, 16), (0, 1, 16), (0, 2, 16)])
    assert p["busy_ns"] == 4  # simultaneous injection/ejection, not their sum
    assert "injection:0" in p["bottleneck_resources"]
    assert "ejection:0" in p["bottleneck_resources"]


def test_real_butterfly_has_stages_and_distinct_hypercube_graph():
    butterfly, cube = NoCSimulator(16, "Butterfly"), NoCSimulator(16, "Hypercube")
    assert butterfly.get_topology_info()["auxiliary_vertices"] == 80
    assert cube.get_topology_info()["auxiliary_vertices"] == 0
    for src in range(16):
        for dst in range(16):
            if src == dst:
                continue
            route = butterfly.get_route(src, dst)
            assert len(route) == 6
            assert all(u >= 16 for u, _ in route[1:])
            assert all(v >= 16 for _, v in route[:-1])
    assert butterfly.simulate_fl_round(150528, "RingAllReduce")["total_bytes"] == 30*150528


def test_clock_scaling_includes_pipeline_and_serialization():
    a = NoCSimulator(16, "Torus", clock_ghz=1).simulate_fl_round(150528, "RingAllReduce")
    b = NoCSimulator(16, "Torus", clock_ghz=2).simulate_fl_round(150528, "RingAllReduce")
    assert a["latency_ns"] == 2*b["latency_ns"]
    assert a["energy_j"] is None
    assert not a["model_assumptions"]["cycle_accurate"]


def test_calibrated_energy_requires_source_and_node():
    with pytest.raises(ValueError, match="source and technology_node"):
        NoCSimulator(4, energy_profile={"pj_per_flit_hop": 1})
    sim = NoCSimulator(2, "Ring", energy_profile={"source": "synthetic unit-test fixture",
                       "technology_node": "not a real process", "pj_per_flit_hop": 2})
    assert sim.simulate_fl_round(17)["energy_j"] == pytest.approx(8e-12)


def test_arbitrary_tile_counts_do_not_create_phantom_nodes():
    assert len(NoCSimulator(3, "Mesh").topology["adjacency"]) == 3
    with pytest.raises(ValueError, match="phantom"):
        NoCSimulator(3, "Mesh", tile_rows=2, tile_cols=2)
    with pytest.raises(ValueError, match="power-of-two"):
        NoCSimulator(3, "Butterfly")
    assert NoCSimulator(1).simulate_fl_round(17, "RingAllReduce")["latency_ns"] == 0


def test_recursive_doubling_volume_and_round_count():
    r = NoCSimulator(16).simulate_fl_round(17, "AllReduce")
    assert r["total_bytes"] == 16*4*17
    assert r["num_phases"] == 4
    with pytest.raises(ValueError, match="power-of-two"):
        NoCSimulator(3).simulate_fl_round(17, "AllReduce")


def test_unsupported_collective_does_not_hide_valid_topologies():
    from src.noc_simulator import compare_topologies_and_protocols
    r = compare_topologies_and_protocols(3, 17)
    assert "error" in r["Butterfly"]["ParameterServer"]
    assert r["Mesh"]["RingAllReduce"]["total_bytes"] == 2*2*17*20
    with pytest.raises(ValueError, match="Unknown protocol"):
        NoCSimulator(1).simulate_fl_round(1, "invalid")


def test_transmit_power_window_tracks_declared_noise():
    from types import SimpleNamespace

    from experiments.link_level import LinkLevelSuite
    cfg = SimpleNamespace(NOISE_POWER_DBM=-90)
    suite = LinkLevelSuite(cfg, {})
    np.testing.assert_equal(suite.RHO_DB+cfg.NOISE_POWER_DBM, np.arange(-10, 31))


def test_preserved_supplied_csi_audit_contains_no_neural_results():
    paths = sorted(Path("results/link_level_corrected").glob("seed_*/link_level_results.json"))
    assert len(paths) == 5
    for path in paths:
        r = json.loads(path.read_text())
        assert r["meta"]["training"]["status"] == "not_run"
        assert not set(r["mean_gain_db"]) & {"fed_ris", "centralized_dl", "local_only"}

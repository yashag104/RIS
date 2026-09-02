import pytest
from src.noc_simulator import compare_topologies_and_protocols, NoCSimulator

def test_noc_topologies_and_protocols():
    results = compare_topologies_and_protocols(
        num_tiles=16,
        model_size_bytes=1024 * 50,  # 50 KB
        num_rounds=5,
        bandwidth_gbps=10.0
    )
    
    topologies = ['Mesh', 'Torus', 'FoldedTorus', 'Tree', 'Butterfly', 'Ring']
    protocols = ['ParameterServer', 'AllReduce', 'RingAllReduce', 'Gossip']
    
    for topo in topologies:
        assert topo in results, f"Missing topology: {topo}"
        assert 'error' not in results[topo], f"Error in {topo}: {results[topo].get('error')}"
        
        for proto in protocols:
            assert proto in results[topo], f"Missing protocol: {proto} in {topo}"
            metrics = results[topo][proto]
            assert 'error' not in metrics, f"Error in {topo}/{proto}: {metrics.get('error')}"
            assert 'total_latency_us' in metrics
            assert 'total_bytes' in metrics

    assert '_rankings' in results

def test_ring_allreduce_optimality():
    sim = NoCSimulator(num_tiles=16, topology="Ring")
    
    res_ps = sim.simulate_fl_round(model_size_bytes=1024, protocol="ParameterServer")
    res_ring = sim.simulate_fl_round(model_size_bytes=1024, protocol="RingAllReduce")
    
    assert res_ring['bottleneck'] == 'longest_ring_link'
    assert res_ps['bottleneck'] == 'server_node'
    assert res_ring['utilization'] >= 0

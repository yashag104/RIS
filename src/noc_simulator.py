"""
Network-on-Chip (NoC) Simulator for Federated Learning on RIS Tiles

Implements discrete-event simulation of FL communication patterns
across different NoC topologies and aggregation protocols.

Topologies: Mesh, Torus, FoldedTorus, Tree, Butterfly, Ring
Protocols: Parameter-Server, All-Reduce, Ring-AllReduce, Gossip

References:
- Dally & Towles, "Principles and Practices of Interconnection Networks," 2004
- Ring-AllReduce: Patarasuk & Yuan, "Bandwidth Optimal All-reduce Algorithms," 2009
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class NoCTopology:
    """
    Builds adjacency graph for various NoC topologies.
    
    Each node represents a tile (processing element).
    Edges represent bidirectional communication links.
    """
    
    @staticmethod
    def build_mesh(rows: int, cols: int) -> Dict:
        """
        2D Mesh topology. Each node connects to up to 4 neighbors.
        Routing: XY deterministic routing.
        """
        num_nodes = rows * cols
        adj = defaultdict(list)
        
        for r in range(rows):
            for c in range(cols):
                node = r * cols + c
                # Right
                if c + 1 < cols:
                    adj[node].append(r * cols + c + 1)
                    adj[r * cols + c + 1].append(node)
                # Down
                if r + 1 < rows:
                    adj[node].append((r + 1) * cols + c)
                    adj[(r + 1) * cols + c].append(node)
        
        # Remove duplicates
        adj = {k: list(set(v)) for k, v in adj.items()}
        
        return {
            'name': 'Mesh',
            'num_nodes': num_nodes,
            'adjacency': dict(adj),
            'rows': rows,
            'cols': cols,
            'bisection_bandwidth': cols,  # Minimum cut
            'diameter': rows + cols - 2,
            'avg_hops': (rows + cols) / 3,  # Approximation
        }
    
    @staticmethod
    def build_torus(rows: int, cols: int) -> Dict:
        """
        2D Torus: Mesh with wrap-around edges.
        Reduces diameter and average hops.
        """
        num_nodes = rows * cols
        adj = defaultdict(list)
        
        for r in range(rows):
            for c in range(cols):
                node = r * cols + c
                # Right (with wrap)
                right = r * cols + (c + 1) % cols
                adj[node].append(right)
                adj[right].append(node)
                # Down (with wrap)
                down = ((r + 1) % rows) * cols + c
                adj[node].append(down)
                adj[down].append(node)
        
        adj = {k: list(set(v)) for k, v in adj.items()}
        
        return {
            'name': 'Torus',
            'num_nodes': num_nodes,
            'adjacency': dict(adj),
            'rows': rows,
            'cols': cols,
            'bisection_bandwidth': 2 * cols,
            'diameter': (rows // 2) + (cols // 2),
            'avg_hops': (rows + cols) / 4,
        }
    
    @staticmethod
    def build_folded_torus(rows: int, cols: int) -> Dict:
        """
        Folded Torus: Torus with halved wrap-around distances.
        Each wrap-around link is the same length as internal links.
        """
        # Same connectivity as torus, but with uniform link lengths
        torus = NoCTopology.build_torus(rows, cols)
        torus['name'] = 'FoldedTorus'
        # In folded torus, all links are equal length (no longer wrap-around penalty)
        torus['diameter'] = max(rows // 2, 1) + max(cols // 2, 1)
        torus['avg_hops'] = (rows / 4 + cols / 4)
        return torus
    
    @staticmethod
    def build_tree(num_nodes: int, branching_factor: int = 2) -> Dict:
        """
        Fat Tree topology. Root is node 0.
        Good for aggregation-heavy traffic (FL).
        """
        adj = defaultdict(list)
        depth = max(1, int(np.ceil(np.log(max(num_nodes, 2)) / np.log(max(branching_factor, 2)))))
        
        for node in range(num_nodes):
            if node == 0:
                continue
            parent = (node - 1) // branching_factor
            if parent < num_nodes:
                adj[node].append(parent)
                adj[parent].append(node)
        
        adj = {k: list(set(v)) for k, v in adj.items()}
        # Ensure all nodes exist
        for n in range(num_nodes):
            if n not in adj:
                adj[n] = []
        
        return {
            'name': 'Tree',
            'num_nodes': num_nodes,
            'adjacency': dict(adj),
            'branching_factor': branching_factor,
            'depth': depth,
            'bisection_bandwidth': branching_factor,
            'diameter': 2 * depth,
            'avg_hops': depth,
        }
    
    @staticmethod
    def build_butterfly(num_nodes: int) -> Dict:
        """
        Butterfly network. Used in FFT-like communication patterns.
        Stages of log(N) with N switches each.
        For simplicity, we model a flattened butterfly (one stage of full crossbar).
        """
        adj = defaultdict(list)
        
        # Flattened butterfly: each node connects to log2(N) other nodes
        # at distances that are powers of 2
        log_n = max(1, int(np.ceil(np.log2(max(num_nodes, 2)))))
        
        for node in range(num_nodes):
            for stage in range(log_n):
                partner = node ^ (1 << stage)  # XOR with 2^stage
                if partner < num_nodes and partner != node:
                    adj[node].append(partner)
                    adj[partner].append(node)
        
        adj = {k: list(set(v)) for k, v in adj.items()}
        for n in range(num_nodes):
            if n not in adj:
                adj[n] = []
        
        return {
            'name': 'Butterfly',
            'num_nodes': num_nodes,
            'adjacency': dict(adj),
            'stages': log_n,
            'bisection_bandwidth': num_nodes // 2,
            'diameter': log_n,
            'avg_hops': log_n / 2,
        }
    
    @staticmethod
    def build_ring(num_nodes: int) -> Dict:
        """
        Ring topology. Optimal for Ring-AllReduce protocol.
        """
        adj = defaultdict(list)
        
        for node in range(num_nodes):
            left = (node - 1) % num_nodes
            right = (node + 1) % num_nodes
            adj[node].extend([left, right])
        
        adj = {k: list(set(v)) for k, v in adj.items()}
        
        return {
            'name': 'Ring',
            'num_nodes': num_nodes,
            'adjacency': dict(adj),
            'bisection_bandwidth': 2,
            'diameter': num_nodes // 2,
            'avg_hops': num_nodes / 4,
        }


class NoCSimulator:
    """
    Discrete-event NoC simulator for FL communication.
    
    Simulates the communication overhead of different FL aggregation
    protocols over various NoC topologies for one FL round.
    """
    
    # Energy model constants
    ENERGY_PER_FLIT_SWITCH = 0.98e-12  # ~1 pJ per flit per switch (45nm)
    ENERGY_PER_FLIT_LINK = 0.37e-12   # ~0.4 pJ per flit per link
    FLIT_SIZE_BYTES = 16               # 128-bit flit
    LINK_LATENCY_NS = 1.0             # 1 ns per hop
    SWITCH_LATENCY_NS = 2.0           # 2 ns router pipeline
    
    def __init__(
        self,
        num_tiles: int,
        topology: str = "Mesh",
        bandwidth_gbps: float = 10.0,
        tile_rows: int = None,
        tile_cols: int = None,
    ):
        """
        Args:
            num_tiles: Number of processing tiles
            topology: One of "Mesh", "Torus", "FoldedTorus", "Tree", "Butterfly", "Ring"
            bandwidth_gbps: Link bandwidth in Gbps
            tile_rows: Grid rows (auto-computed if None)
            tile_cols: Grid cols (auto-computed if None)
        """
        self.num_tiles = num_tiles
        self.topology_name = topology
        self.bandwidth_gbps = bandwidth_gbps
        self.bytes_per_sec = bandwidth_gbps * 1e9 / 8
        
        # Compute grid dimensions for 2D topologies
        if tile_rows is None or tile_cols is None:
            sqrt_n = int(np.ceil(np.sqrt(num_tiles)))
            tile_rows = sqrt_n
            tile_cols = max(1, (num_tiles + sqrt_n - 1) // sqrt_n)
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        
        # Build topology
        self.topology = self._build_topology(topology)
        
        # Precompute shortest paths using BFS
        self.shortest_paths = self._compute_shortest_paths()
    
    def _build_topology(self, name: str) -> Dict:
        """Build the specified topology."""
        builders = {
            'Mesh': lambda: NoCTopology.build_mesh(self.tile_rows, self.tile_cols),
            'Torus': lambda: NoCTopology.build_torus(self.tile_rows, self.tile_cols),
            'FoldedTorus': lambda: NoCTopology.build_folded_torus(self.tile_rows, self.tile_cols),
            'Tree': lambda: NoCTopology.build_tree(self.num_tiles),
            'Butterfly': lambda: NoCTopology.build_butterfly(self.num_tiles),
            'Ring': lambda: NoCTopology.build_ring(self.num_tiles),
        }
        if name not in builders:
            raise ValueError(f"Unknown topology: {name}. Options: {list(builders.keys())}")
        return builders[name]()
    
    def _compute_shortest_paths(self) -> Dict:
        """BFS-based all-pairs shortest path computation."""
        paths = {}
        adj = self.topology['adjacency']
        
        for src in range(self.num_tiles):
            dist = {src: 0}
            queue = [src]
            idx = 0
            while idx < len(queue):
                node = queue[idx]
                idx += 1
                for neighbor in adj.get(node, []):
                    if neighbor not in dist:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
            paths[src] = dist
        
        return paths
    
    def get_hop_count(self, src: int, dst: int) -> int:
        """Get hop count between two nodes."""
        return self.shortest_paths.get(src, {}).get(dst, self.num_tiles)  # Fallback
    
    def simulate_fl_round(
        self,
        model_size_bytes: int,
        protocol: str = "ParameterServer",
    ) -> Dict:
        """
        Simulate one FL round of communication.
        
        Args:
            model_size_bytes: Size of model parameters in bytes
            protocol: "ParameterServer", "AllReduce", "RingAllReduce", "Gossip"
            
        Returns:
            Dictionary with communication metrics for this round
        """
        protocols = {
            'ParameterServer': self._simulate_parameter_server,
            'AllReduce': self._simulate_all_reduce,
            'RingAllReduce': self._simulate_ring_allreduce,
            'Gossip': self._simulate_gossip,
        }
        
        if protocol not in protocols:
            raise ValueError(f"Unknown protocol: {protocol}. Options: {list(protocols.keys())}")
        
        return protocols[protocol](model_size_bytes)
    
    def _simulate_parameter_server(self, model_size_bytes: int) -> Dict:
        """
        Parameter Server protocol: all tiles send to node 0, node 0 broadcasts back.
        
        Traffic pattern: star centered at node 0.
        """
        N = self.num_tiles
        server_node = 0
        
        # Phase 1: Upload (all tiles → server)
        upload_hops = []
        for tile in range(N):
            if tile != server_node:
                hops = self.get_hop_count(tile, server_node)
                upload_hops.append(hops)
        
        # Phase 2: Download (server → all tiles)
        download_hops = upload_hops.copy()  # Symmetric
        
        total_hops = sum(upload_hops) + sum(download_hops)
        
        # Bottleneck: server must receive N-1 models sequentially 
        # (link bandwidth limited at server)
        num_flits = max(1, model_size_bytes // self.FLIT_SIZE_BYTES)
        max_upload_hops = max(upload_hops) if upload_hops else 1
        
        # Latency = max path × per-hop latency + serialization
        serialization_ns = (num_flits * self.FLIT_SIZE_BYTES * 8) / (self.bandwidth_gbps * 1e9) * 1e9
        
        # Upload bottleneck: N-1 models through server's ports
        upload_latency_ns = (N - 1) * serialization_ns + max_upload_hops * (
            self.LINK_LATENCY_NS + self.SWITCH_LATENCY_NS)
        download_latency_ns = serialization_ns + max_upload_hops * (
            self.LINK_LATENCY_NS + self.SWITCH_LATENCY_NS)
        total_latency_ns = upload_latency_ns + download_latency_ns
        
        # Total bytes transferred
        total_bytes = 2 * (N - 1) * model_size_bytes
        
        # Energy
        total_energy = total_hops * num_flits * (
            self.ENERGY_PER_FLIT_SWITCH + self.ENERGY_PER_FLIT_LINK)
        
        # Bandwidth utilization (per-link, not aggregate)
        time_sec = total_latency_ns * 1e-9
        achieved_throughput = total_bytes / max(time_sec, 1e-15)
        # Server has limited ports; concurrent links = min(N-1, server_degree)
        server_degree = len(self.topology.get('adjacency', {}).get(0, [1]))
        num_active_links = max(min(N - 1, server_degree), 1)
        utilization = min(achieved_throughput / (num_active_links * self.bytes_per_sec), 1.0)
        
        # Congestion: server node is bottleneck
        server_load = 2 * (N - 1)  # Incoming + outgoing messages
        max_link_load = server_load
        avg_link_load = total_hops / max(N, 1)
        congestion_ratio = max_link_load / max(avg_link_load, 1)
        
        return {
            'protocol': 'ParameterServer',
            'topology': self.topology_name,
            'total_bytes': total_bytes,
            'total_hops': total_hops,
            'latency_ns': total_latency_ns,
            'latency_us': total_latency_ns / 1000,
            'energy_j': total_energy,
            'energy_nj': total_energy * 1e9,
            'utilization': utilization,
            'congestion_ratio': congestion_ratio,
            'bottleneck': 'server_node',
            'num_phases': 2,
        }
    
    def _simulate_all_reduce(self, model_size_bytes: int) -> Dict:
        """
        All-Reduce: reduce-to-all pattern.
        Phase 1: Reduce (tree reduction to root)
        Phase 2: Broadcast (root to all)
        """
        N = self.num_tiles
        
        # Tree-based all-reduce
        depth = max(1, int(np.ceil(np.log2(max(N, 2)))))
        
        # Reduce phase: log2(N) stages, each with N/2 communications
        reduce_hops_per_stage = []
        for stage in range(depth):
            stride = 1 << stage
            stage_hops = 0
            count = 0
            for node in range(N):
                partner = node ^ stride
                if partner < N and partner > node:
                    hops = self.get_hop_count(node, partner)
                    stage_hops += hops
                    count += 1
            reduce_hops_per_stage.append(stage_hops)
        
        total_reduce_hops = sum(reduce_hops_per_stage)
        total_broadcast_hops = total_reduce_hops  # Symmetric
        total_hops = total_reduce_hops + total_broadcast_hops
        
        num_flits = max(1, model_size_bytes // self.FLIT_SIZE_BYTES)
        serialization_ns = (num_flits * self.FLIT_SIZE_BYTES * 8) / (self.bandwidth_gbps * 1e9) * 1e9
        
        # Each stage has one serialization + hop latency (pipelined)
        max_hop_per_stage = max(
            max(self.get_hop_count(n, n ^ (1 << s))
                for n in range(N) if (n ^ (1 << s)) < N)
            for s in range(depth)
        ) if depth > 0 else 1
        
        per_stage_latency = serialization_ns + max_hop_per_stage * (
            self.LINK_LATENCY_NS + self.SWITCH_LATENCY_NS)
        total_latency_ns = 2 * depth * per_stage_latency
        
        total_bytes = 2 * depth * (N // 2) * model_size_bytes
        
        total_energy = total_hops * num_flits * (
            self.ENERGY_PER_FLIT_SWITCH + self.ENERGY_PER_FLIT_LINK)
        
        time_sec = total_latency_ns * 1e-9
        achieved_throughput = total_bytes / max(time_sec, 1e-15)
        # In each butterfly stage, N/2 pairs communicate simultaneously
        num_active_links = max(N // 2, 1)
        utilization = min(achieved_throughput / (num_active_links * self.bytes_per_sec), 1.0)

        return {
            'protocol': 'AllReduce',
            'topology': self.topology_name,
            'total_bytes': total_bytes,
            'total_hops': total_hops,
            'latency_ns': total_latency_ns,
            'latency_us': total_latency_ns / 1000,
            'energy_j': total_energy,
            'energy_nj': total_energy * 1e9,
            'utilization': utilization,
            'congestion_ratio': 1.0 + 0.1 * depth,
            'bottleneck': 'tree_root',
            'num_phases': 2 * depth,
        }
    
    def _simulate_ring_allreduce(self, model_size_bytes: int) -> Dict:
        """
        Ring-AllReduce: bandwidth-optimal all-reduce.
        Phase 1: Scatter-Reduce (N-1 steps around ring)
        Phase 2: All-Gather (N-1 steps around ring)
        
        Each step transmits model_size / N bytes.
        """
        N = self.num_tiles
        
        # In ring-allreduce, data is chunked into N segments
        chunk_size = max(1, model_size_bytes // N)
        num_flits_per_chunk = max(1, chunk_size // self.FLIT_SIZE_BYTES)
        
        # Each step: every node sends one chunk to its ring neighbor
        # Total steps: 2 * (N - 1)
        num_steps = 2 * (N - 1)
        
        # For ring topology, each hop is 1. For other topologies,
        # we route along the Hamiltonian path
        ring_hops = []
        for node in range(N):
            next_node = (node + 1) % N
            hops = self.get_hop_count(node, next_node)
            ring_hops.append(hops)
        
        avg_ring_hop = np.mean(ring_hops)
        max_ring_hop = max(ring_hops)
        
        # Total hops: num_steps × N concurrent transfers × avg_hops
        total_hops = num_steps * N * avg_ring_hop
        
        # Latency: pipelined, so = num_steps × (serialization + max_hop)
        serialization_ns = (num_flits_per_chunk * self.FLIT_SIZE_BYTES * 8) / (
            self.bandwidth_gbps * 1e9) * 1e9
        per_step_latency = serialization_ns + max_ring_hop * (
            self.LINK_LATENCY_NS + self.SWITCH_LATENCY_NS)
        total_latency_ns = num_steps * per_step_latency
        
        # Total bytes: 2 * (N-1) * chunk_size * N  (but bandwidth-optimal)
        # Actually: 2 * (N-1)/N * model_size ≈ 2 * model_size for large N
        total_bytes = 2 * (N - 1) * chunk_size * N
        
        total_energy = total_hops * num_flits_per_chunk * (
            self.ENERGY_PER_FLIT_SWITCH + self.ENERGY_PER_FLIT_LINK)
        
        time_sec = total_latency_ns * 1e-9
        achieved_throughput = total_bytes / max(time_sec, 1e-15)
        # All N nodes send to ring neighbor simultaneously
        num_active_links = N
        utilization = min(achieved_throughput / (num_active_links * self.bytes_per_sec), 1.0)

        return {
            'protocol': 'RingAllReduce',
            'topology': self.topology_name,
            'total_bytes': total_bytes,
            'total_hops': int(total_hops),
            'latency_ns': total_latency_ns,
            'latency_us': total_latency_ns / 1000,
            'energy_j': total_energy,
            'energy_nj': total_energy * 1e9,
            'utilization': utilization,
            'congestion_ratio': max_ring_hop / max(avg_ring_hop, 1e-10),
            'bottleneck': 'longest_ring_link',
            'num_phases': num_steps,
            'chunk_size_bytes': chunk_size,
        }
    
    def _simulate_gossip(self, model_size_bytes: int) -> Dict:
        """
        Gossip protocol: each node randomly selects a peer and exchanges models.
        
        Each round: N/2 peer exchanges (non-overlapping pairs).
        Need O(log N) gossip rounds for convergence.
        """
        N = self.num_tiles
        
        # Number of gossip rounds for mixing
        num_gossip_rounds = max(1, int(np.ceil(np.log2(max(N, 2)))) + 1)
        
        # Each gossip round: N/2 random peer exchanges
        total_hops = 0
        max_hops_per_round = 0
        
        np.random.seed(42)  # Reproducible
        for _ in range(num_gossip_rounds):
            nodes = list(range(N))
            np.random.shuffle(nodes)
            round_hops = 0
            round_max = 0
            pairs = 0
            
            for i in range(0, len(nodes) - 1, 2):
                n1, n2 = nodes[i], nodes[i + 1]
                hops = self.get_hop_count(n1, n2)
                round_hops += 2 * hops  # Bidirectional exchange
                round_max = max(round_max, hops)
                pairs += 1
            
            total_hops += round_hops
            max_hops_per_round = max(max_hops_per_round, round_max)
        
        num_flits = max(1, model_size_bytes // self.FLIT_SIZE_BYTES)
        serialization_ns = (num_flits * self.FLIT_SIZE_BYTES * 8) / (
            self.bandwidth_gbps * 1e9) * 1e9
        
        # Each gossip round: pairs exchange simultaneously (parallel)
        per_round_latency = serialization_ns + max_hops_per_round * (
            self.LINK_LATENCY_NS + self.SWITCH_LATENCY_NS)
        total_latency_ns = num_gossip_rounds * per_round_latency
        
        # Total bytes: num_gossip_rounds × N × model_size (each node sends once per round)
        total_bytes = num_gossip_rounds * N * model_size_bytes
        
        total_energy = total_hops * num_flits * (
            self.ENERGY_PER_FLIT_SWITCH + self.ENERGY_PER_FLIT_LINK)
        
        time_sec = total_latency_ns * 1e-9
        achieved_throughput = total_bytes / max(time_sec, 1e-15)
        # N/2 pairs exchange simultaneously per gossip round
        num_active_links = max(N // 2, 1)
        utilization = min(achieved_throughput / (num_active_links * self.bytes_per_sec), 1.0)

        return {
            'protocol': 'Gossip',
            'topology': self.topology_name,
            'total_bytes': total_bytes,
            'total_hops': total_hops,
            'latency_ns': total_latency_ns,
            'latency_us': total_latency_ns / 1000,
            'energy_j': total_energy,
            'energy_nj': total_energy * 1e9,
            'utilization': utilization,
            'congestion_ratio': 1.0,  # Load is distributed
            'bottleneck': 'mixing_time',
            'num_phases': num_gossip_rounds,
            'gossip_rounds': num_gossip_rounds,
        }
    
    def simulate_full_fl_training(
        self,
        model_size_bytes: int,
        num_rounds: int,
        protocol: str = "ParameterServer"
    ) -> Dict:
        """
        Simulate communication for a full FL training session.
        
        Args:
            model_size_bytes: Size of model in bytes
            num_rounds: Number of FL rounds
            protocol: Communication protocol
            
        Returns:
            Aggregated metrics over all rounds
        """
        round_metrics = self.simulate_fl_round(model_size_bytes, protocol)
        
        return {
            'protocol': protocol,
            'topology': self.topology_name,
            'num_rounds': num_rounds,
            'per_round_bytes': round_metrics['total_bytes'],
            'per_round_latency_us': round_metrics['latency_us'],
            'per_round_energy_nj': round_metrics['energy_nj'],
            'total_bytes': round_metrics['total_bytes'] * num_rounds,
            'total_latency_us': round_metrics['latency_us'] * num_rounds,
            'total_latency_ms': round_metrics['latency_us'] * num_rounds / 1000,
            'total_energy_nj': round_metrics['energy_nj'] * num_rounds,
            'total_energy_uj': round_metrics['energy_nj'] * num_rounds / 1000,
            'avg_utilization': round_metrics['utilization'],
            'congestion_ratio': round_metrics['congestion_ratio'],
            'topology_diameter': self.topology.get('diameter', -1),
            'topology_bisection_bw': self.topology.get('bisection_bandwidth', -1),
        }
    
    def get_topology_info(self) -> Dict:
        """Return topology properties."""
        return {
            'name': self.topology_name,
            'num_tiles': self.num_tiles,
            'diameter': self.topology.get('diameter', -1),
            'bisection_bandwidth': self.topology.get('bisection_bandwidth', -1),
            'avg_hops': self.topology.get('avg_hops', -1),
            'num_links': sum(len(v) for v in self.topology['adjacency'].values()) // 2,
            'degree': max(len(v) for v in self.topology['adjacency'].values()) if self.topology['adjacency'] else 0,
        }


def compare_topologies_and_protocols(
    num_tiles: int,
    model_size_bytes: int,
    num_rounds: int = 20,
    bandwidth_gbps: float = 10.0,
) -> Dict:
    """
    Comprehensive comparison of all topologies × all protocols.
    
    Returns:
        Nested dictionary: topology → protocol → metrics
    """
    topologies = ['Mesh', 'Torus', 'FoldedTorus', 'Tree', 'Butterfly', 'Ring']
    protocols = ['ParameterServer', 'AllReduce', 'RingAllReduce', 'Gossip']
    
    results = {}
    
    for topo in topologies:
        results[topo] = {}
        try:
            sim = NoCSimulator(num_tiles, topology=topo, bandwidth_gbps=bandwidth_gbps)
            results[topo]['_info'] = sim.get_topology_info()
            
            for proto in protocols:
                try:
                    metrics = sim.simulate_full_fl_training(
                        model_size_bytes, num_rounds, proto)
                    results[topo][proto] = metrics
                except Exception as e:
                    results[topo][proto] = {'error': str(e)}
        except Exception as e:
            results[topo] = {'error': str(e)}
    
    # Rank topologies by total latency for each protocol
    rankings = {}
    for proto in protocols:
        latencies = {}
        for topo in topologies:
            if proto in results.get(topo, {}) and 'total_latency_us' in results[topo].get(proto, {}):
                latencies[topo] = results[topo][proto]['total_latency_us']
        
        if latencies:
            ranked = sorted(latencies.items(), key=lambda x: x[1])
            rankings[proto] = {
                'best': ranked[0][0],
                'worst': ranked[-1][0],
                'ranking': [t[0] for t in ranked],
                'latencies': {t[0]: t[1] for t in ranked},
            }
    
    results['_rankings'] = rankings
    return results

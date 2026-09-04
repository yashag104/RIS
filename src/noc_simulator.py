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

from collections import defaultdict

import numpy as np

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
    def build_mesh(rows: int, cols: int) -> dict:
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
    def build_torus(rows: int, cols: int) -> dict:
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
    def build_folded_torus(rows: int, cols: int) -> dict:
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
    def build_tree(num_nodes: int, branching_factor: int = 2) -> dict:
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
    def build_butterfly(num_nodes: int) -> dict:
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
    def build_ring(num_nodes: int) -> dict:
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
        tile_rows: int | None = None,
        tile_cols: int | None = None,
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
    
    def _build_topology(self, name: str) -> dict:
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
    
    def _compute_shortest_paths(self) -> dict:
        """BFS-based all-pairs shortest path computation.

        Also records the BFS predecessor tree so that a single deterministic
        shortest route can be reconstructed for every source/destination pair.
        """
        paths = {}
        self._predecessors = {}
        adj = self.topology['adjacency']

        for src in range(self.num_tiles):
            dist = {src: 0}
            pred = {src: None}
            queue = [src]
            idx = 0
            while idx < len(queue):
                node = queue[idx]
                idx += 1
                for neighbor in sorted(adj.get(node, [])):
                    if neighbor not in dist:
                        dist[neighbor] = dist[node] + 1
                        pred[neighbor] = node
                        queue.append(neighbor)
            paths[src] = dist
            self._predecessors[src] = pred

        return paths

    def get_hop_count(self, src: int, dst: int) -> int:
        """Get hop count between two nodes."""
        return self.shortest_paths.get(src, {}).get(dst, self.num_tiles)  # Fallback

    def get_route(self, src: int, dst: int) -> list:
        """Return the deterministic shortest route as a list of directed links.

        Routing is minimal and deterministic (BFS tree order), matching the
        dimension-ordered routing assumed for the mesh/torus fabrics.
        """
        pred = self._predecessors.get(src, {})
        if dst not in pred:
            return []
        links = []
        node = dst
        while node != src:
            prev = pred[node]
            links.append((prev, node))
            node = prev
        links.reverse()
        return links

    def _phase_cost(self, transfers: list) -> dict:
        """Cost of one communication phase under static link-contention analysis.

        Args:
            transfers: list of (src, dst, bytes) tuples that are injected
                concurrently in this phase.

        Returns:
            Dict with the phase latency, the time the bottleneck link spends
            serialising, the per-link byte loads and the total flit-hops.

        The phase latency is set by the most heavily loaded directed link
        (bytes on that link / link rate) plus the traversal latency of the
        longest route. Because the serialisation term is one addend of that
        sum, the busy fraction it defines can never exceed one.
        """
        link_bytes = defaultdict(float)
        max_hops = 0
        flit_hops = 0.0

        for src, dst, nbytes in transfers:
            route = self.get_route(src, dst)
            if not route:
                continue
            max_hops = max(max_hops, len(route))
            n_flits = max(1, int(nbytes) // self.FLIT_SIZE_BYTES)
            flit_hops += n_flits * len(route)
            for link in route:
                link_bytes[link] += nbytes

        if not link_bytes:
            return {'latency_ns': 0.0, 'busy_ns': 0.0, 'link_bytes': {},
                    'bottleneck_bytes': 0.0, 'flit_hops': 0.0, 'max_hops': 0}

        bottleneck_bytes = max(link_bytes.values())
        # ns = bits / (Gbit/s), since 1 Gbit/s = 1 bit/ns
        busy_ns = bottleneck_bytes * 8 / self.bandwidth_gbps
        traversal_ns = max_hops * (self.LINK_LATENCY_NS + self.SWITCH_LATENCY_NS)

        return {
            'latency_ns': busy_ns + traversal_ns,
            'busy_ns': busy_ns,
            'link_bytes': dict(link_bytes),
            'bottleneck_bytes': bottleneck_bytes,
            'flit_hops': flit_hops,
            'max_hops': max_hops,
        }

    @staticmethod
    def _combine_phases(phases: list) -> dict:
        """Aggregate sequential phases into round-level latency and utilization."""
        total_latency_ns = sum(p['latency_ns'] for p in phases)
        total_busy_ns = sum(p['busy_ns'] for p in phases)
        flit_hops = sum(p['flit_hops'] for p in phases)

        merged = defaultdict(float)
        for p in phases:
            for link, nbytes in p['link_bytes'].items():
                merged[link] += nbytes

        loads = list(merged.values()) or [0.0]
        max_load = max(loads)
        mean_load = sum(loads) / len(loads)

        return {
            'total_latency_ns': total_latency_ns,
            'utilization': (total_busy_ns / total_latency_ns) if total_latency_ns > 0 else 0.0,
            'flit_hops': flit_hops,
            'bottleneck_link_bytes': max_load,
            'congestion_ratio': (max_load / mean_load) if mean_load > 0 else 1.0,
            'num_links_used': len(merged),
        }

    
    def simulate_fl_round(
        self,
        model_size_bytes: int,
        protocol: str = "ParameterServer",
    ) -> dict:
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
    
    def _simulate_parameter_server(self, model_size_bytes: int) -> dict:
        """
        Parameter Server protocol: all tiles send to node 0, node 0 broadcasts back.

        Traffic pattern: star centered at node 0. The links incident to the
        server carry every model, so they are the bottleneck.
        """
        N = self.num_tiles
        server = 0

        upload = [(t, server, model_size_bytes) for t in range(N) if t != server]
        download = [(server, t, model_size_bytes) for t in range(N) if t != server]

        phases = [self._phase_cost(upload), self._phase_cost(download)]
        agg = self._combine_phases(phases)

        total_bytes = 2 * (N - 1) * model_size_bytes
        total_energy = agg['flit_hops'] * (
            self.ENERGY_PER_FLIT_SWITCH + self.ENERGY_PER_FLIT_LINK)

        return {
            'protocol': 'ParameterServer',
            'topology': self.topology_name,
            'total_bytes': total_bytes,
            'total_hops': int(sum(len(self.get_route(s, d)) for s, d, _ in upload + download)),
            'latency_ns': agg['total_latency_ns'],
            'latency_us': agg['total_latency_ns'] / 1000,
            'energy_j': total_energy,
            'energy_nj': total_energy * 1e9,
            'utilization': agg['utilization'],
            'aggregate_throughput_gbps': (total_bytes * 8) / max(agg['total_latency_ns'], 1e-9),
            'bottleneck_link_bytes': agg['bottleneck_link_bytes'],
            'congestion_ratio': agg['congestion_ratio'],
            'bottleneck': 'server_node',
            'num_phases': 2,
        }

    def _simulate_all_reduce(self, model_size_bytes: int) -> dict:
        """
        Recursive-halving/doubling All-Reduce.
        Phase 1: Reduce over log2(N) butterfly stages.
        Phase 2: Broadcast back over the same stages in reverse.
        """
        N = self.num_tiles
        depth = max(1, int(np.ceil(np.log2(max(N, 2)))))

        phases = []
        total_bytes = 0
        for stage in range(depth):
            stride = 1 << stage
            transfers = [(node, node ^ stride, model_size_bytes)
                         for node in range(N)
                         if node ^ stride < N and node ^ stride > node]
            if not transfers:
                continue
            cost = self._phase_cost(transfers)
            phases.append(cost)
            phases.append(cost)  # symmetric broadcast stage
            total_bytes += 2 * len(transfers) * model_size_bytes

        agg = self._combine_phases(phases)
        total_energy = agg['flit_hops'] * (
            self.ENERGY_PER_FLIT_SWITCH + self.ENERGY_PER_FLIT_LINK)

        return {
            'protocol': 'AllReduce',
            'topology': self.topology_name,
            'total_bytes': total_bytes,
            'total_hops': int(agg['flit_hops'] * self.FLIT_SIZE_BYTES / max(model_size_bytes, 1)),
            'latency_ns': agg['total_latency_ns'],
            'latency_us': agg['total_latency_ns'] / 1000,
            'energy_j': total_energy,
            'energy_nj': total_energy * 1e9,
            'utilization': agg['utilization'],
            'aggregate_throughput_gbps': (total_bytes * 8) / max(agg['total_latency_ns'], 1e-9),
            'bottleneck_link_bytes': agg['bottleneck_link_bytes'],
            'congestion_ratio': agg['congestion_ratio'],
            'bottleneck': 'butterfly_stage',
            'num_phases': len(phases),
        }

    def _simulate_ring_allreduce(self, model_size_bytes: int) -> dict:
        """
        Ring-AllReduce: bandwidth-optimal all-reduce.
        Phase 1: Scatter-Reduce (N-1 steps around the ring).
        Phase 2: All-Gather (N-1 steps around the ring).

        Each step every node forwards one model_size/N chunk to its ring
        successor, so the per-step payload is independent of N.
        """
        N = self.num_tiles
        chunk_size = max(1, model_size_bytes // max(N, 1))
        num_steps = 2 * max(N - 1, 1)

        step_transfers = [(node, (node + 1) % N, chunk_size)
                          for node in range(N)] if N > 1 else []
        step_cost = self._phase_cost(step_transfers)
        phases = [step_cost] * num_steps
        agg = self._combine_phases(phases)

        total_bytes = num_steps * N * chunk_size
        total_energy = agg['flit_hops'] * (
            self.ENERGY_PER_FLIT_SWITCH + self.ENERGY_PER_FLIT_LINK)

        return {
            'protocol': 'RingAllReduce',
            'topology': self.topology_name,
            'total_bytes': total_bytes,
            'total_hops': int(num_steps * sum(
                len(self.get_route(n, (n + 1) % N)) for n in range(N))),
            'latency_ns': agg['total_latency_ns'],
            'latency_us': agg['total_latency_ns'] / 1000,
            'energy_j': total_energy,
            'energy_nj': total_energy * 1e9,
            'utilization': agg['utilization'],
            'aggregate_throughput_gbps': (total_bytes * 8) / max(agg['total_latency_ns'], 1e-9),
            'bottleneck_link_bytes': agg['bottleneck_link_bytes'],
            'congestion_ratio': agg['congestion_ratio'],
            'bottleneck': 'busiest_ring_link',
            'num_phases': num_steps,
            'chunk_size_bytes': chunk_size,
        }

    def _simulate_gossip(self, model_size_bytes: int) -> dict:
        """
        Gossip protocol: each node exchanges models with a random peer.

        Each gossip round pairs the nodes at random; O(log N) rounds are needed
        for mixing. Pairs are spatially arbitrary, so multi-hop routes collide
        on shared links and the contention analysis captures that cost.
        """
        N = self.num_tiles
        num_gossip_rounds = max(1, int(np.ceil(np.log2(max(N, 2)))) + 1)

        rng = np.random.default_rng(42)  # Reproducible
        phases = []
        total_bytes = 0
        for _ in range(num_gossip_rounds):
            nodes = list(range(N))
            rng.shuffle(nodes)
            transfers = []
            for i in range(0, len(nodes) - 1, 2):
                n1, n2 = int(nodes[i]), int(nodes[i + 1])
                transfers.append((n1, n2, model_size_bytes))
                transfers.append((n2, n1, model_size_bytes))
            if transfers:
                phases.append(self._phase_cost(transfers))
                total_bytes += len(transfers) * model_size_bytes

        agg = self._combine_phases(phases)
        total_energy = agg['flit_hops'] * (
            self.ENERGY_PER_FLIT_SWITCH + self.ENERGY_PER_FLIT_LINK)

        return {
            'protocol': 'Gossip',
            'topology': self.topology_name,
            'total_bytes': total_bytes,
            'total_hops': int(agg['flit_hops'] * self.FLIT_SIZE_BYTES / max(model_size_bytes, 1)),
            'latency_ns': agg['total_latency_ns'],
            'latency_us': agg['total_latency_ns'] / 1000,
            'energy_j': total_energy,
            'energy_nj': total_energy * 1e9,
            'utilization': agg['utilization'],
            'aggregate_throughput_gbps': (total_bytes * 8) / max(agg['total_latency_ns'], 1e-9),
            'bottleneck_link_bytes': agg['bottleneck_link_bytes'],
            'congestion_ratio': agg['congestion_ratio'],
            'bottleneck': 'mixing_time',
            'num_phases': num_gossip_rounds,
            'gossip_rounds': num_gossip_rounds,
        }

    def simulate_full_fl_training(
        self,
        model_size_bytes: int,
        num_rounds: int,
        protocol: str = "ParameterServer"
    ) -> dict:
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
            'aggregate_throughput_gbps': round_metrics['aggregate_throughput_gbps'],
            'congestion_ratio': round_metrics['congestion_ratio'],
            'topology_diameter': self.topology.get('diameter', -1),
            'topology_bisection_bw': self.topology.get('bisection_bandwidth', -1),
        }
    
    def get_topology_info(self) -> dict:
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
) -> dict:
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

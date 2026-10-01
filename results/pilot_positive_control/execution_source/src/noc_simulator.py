"""Analytical traffic/serialization model for tile-controller interconnects.

This is not a discrete-event or cycle-accurate simulator. Each phase costs the
maximum directed-link or endpoint serialization time plus longest-path pipeline
delay. It omits buffers, arbitration, credit stalls, packet headers, reduction
computation, and clock/physical-design closure. Energy is unknown unless a caller
supplies an explicitly sourced calibration. A physical RIS panel is not a die.
"""
import importlib.util
import math
from collections import defaultdict, deque

import numpy as np

HAS_NETWORKX = importlib.util.find_spec("networkx") is not None


def _graph(name, n, edges, directed=False, **extra):
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        if u == v:
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set())
        if not directed:
            adj[v].add(u)
    return dict(name=name, num_nodes=n, adjacency={k: sorted(v) for k, v in adj.items()},
                directed=directed, **extra)


class NoCTopology:
    """Logical graphs; auxiliary butterfly stage vertices are not tile clients."""

    @staticmethod
    def build_mesh(rows, cols):
        edges = []
        for r in range(rows):
            for c in range(cols):
                u = r*cols+c
                if c+1 < cols:
                    edges.append((u, u+1))
                if r+1 < rows:
                    edges.append((u, u+cols))
        return _graph("Mesh", rows*cols, edges, rows=rows, cols=cols)

    @staticmethod
    def build_torus(rows, cols):
        edges = [(r*cols+c, r*cols+(c+1) % cols) for r in range(rows) for c in range(cols)]
        edges += [(r*cols+c, ((r+1) % rows)*cols+c) for r in range(rows) for c in range(cols)]
        return _graph("Torus", rows*cols, edges, rows=rows, cols=cols)

    @staticmethod
    def build_folded_torus(rows, cols):
        g = NoCTopology.build_torus(rows, cols)
        g.update(name="FoldedTorus", physical_layout="not modeled; same logical graph as torus")
        return g

    @staticmethod
    def build_tree(num_nodes, branching_factor=2):
        if branching_factor < 2:
            raise ValueError("Tree branching factor must be at least two")
        return _graph("Tree", num_nodes, [((n-1)//branching_factor, n) for n in range(1, num_nodes)],
                      description="binary heap tree, not fat tree", branching_factor=branching_factor)

    @staticmethod
    def build_hypercube(num_nodes):
        if num_nodes < 1 or num_nodes & (num_nodes-1):
            raise ValueError("Hypercube requires a power-of-two endpoint count")
        return _graph("Hypercube", num_nodes,
                      [(n, n ^ (1 << s)) for n in range(num_nodes)
                       for s in range(num_nodes.bit_length()-1)])

    @staticmethod
    def build_butterfly(num_nodes):
        """Directed radix-2 butterfly graph with explicit input/output stages.

        Level s has N vertices; each vertex connects to row i and i xor 2**s
        in level s+1. A tile injects at its input row and receives at its output
        row. Stage vertices are extra resources (N*(log2(N)+1)), so this is
        not an equal-area comparison to direct topologies. This graph replaces
        the old mislabeled XOR hypercube, which remains available by name.
        """
        if num_nodes < 1 or num_nodes & (num_nodes-1):
            raise ValueError("Butterfly requires a power-of-two endpoint count")
        depth = num_nodes.bit_length()-1
        def node(stage, row):
            return num_nodes + stage*num_nodes + row
        edges = [(i, node(0, i)) for i in range(num_nodes)]
        edges += [(node(depth, i), i) for i in range(num_nodes)]
        for stage in range(depth):
            for i in range(num_nodes):
                edges += [(node(stage, i), node(stage+1, i)),
                          (node(stage, i), node(stage+1, i ^ (1 << stage)))]
        return _graph("Butterfly", num_nodes, edges, directed=True,
                      stages=depth, auxiliary_vertices=num_nodes*(depth+1),
                      description="directed radix-2 butterfly; distinct stage resources")

    @staticmethod
    def build_ring(num_nodes):
        return _graph("Ring", num_nodes, [(i, (i+1) % num_nodes) for i in range(num_nodes)])


class NoCSimulator:
    """Compatibility name for an analytical, full-duplex traffic model.

    Default scenario: 128 bits/cycle at 1 GHz, one injection and one ejection
    port per endpoint at the link rate, one link plus two router pipeline
    cycles per hop. These are assumptions, not synthesized technology results.
    """
    FLIT_SIZE_BYTES = 16

    def __init__(self, num_tiles, topology="Mesh", bandwidth_gbps=None,
                 tile_rows=None, tile_cols=None, *, link_width_bits=128,
                 clock_ghz=1.0, endpoint_bandwidth_gbps=None, router_cycles=2,
                 link_cycles=1, energy_profile=None):
        if num_tiles < 1 or int(num_tiles) != num_tiles:
            raise ValueError("Positive integer tile count required")
        if link_width_bits <= 0 or link_width_bits % 8 or clock_ghz <= 0:
            raise ValueError("Positive byte-aligned width and clock required")
        self.num_tiles, self.topology_name = num_tiles, topology
        self.link_width_bits, self.clock_ghz = link_width_bits, clock_ghz
        self.FLIT_SIZE_BYTES = link_width_bits//8
        self.bandwidth_gbps = float(bandwidth_gbps if bandwidth_gbps is not None else link_width_bits*clock_ghz)
        self.endpoint_bandwidth_gbps = float(self.bandwidth_gbps if endpoint_bandwidth_gbps is None else endpoint_bandwidth_gbps)
        if min(self.bandwidth_gbps, self.endpoint_bandwidth_gbps) <= 0 or min(router_cycles, link_cycles) < 0:
            raise ValueError("Positive bandwidth and nonnegative pipeline cycles required")
        self.bytes_per_sec = self.bandwidth_gbps*1e9/8
        self.router_cycles, self.link_cycles = router_cycles, link_cycles
        self.energy_profile = energy_profile
        if energy_profile is not None:
            if not energy_profile.get("source") or not energy_profile.get("technology_node"):
                raise ValueError("Energy calibration requires source and technology_node")
            if energy_profile.get("pj_per_flit_hop", -1) < 0:
                raise ValueError("Energy calibration requires nonnegative pj_per_flit_hop")
        if tile_rows is None and tile_cols is None:
            tile_rows = math.isqrt(num_tiles)
            while num_tiles % tile_rows:
                tile_rows -= 1
            tile_cols = num_tiles//tile_rows
        elif tile_rows is None:
            tile_rows = num_tiles//tile_cols
        elif tile_cols is None:
            tile_cols = num_tiles//tile_rows
        if tile_rows*tile_cols != num_tiles:
            raise ValueError("Grid dimensions must equal tile count; no phantom routers")
        self.tile_rows, self.tile_cols = tile_rows, tile_cols
        builders = {"Mesh": lambda: NoCTopology.build_mesh(tile_rows, tile_cols),
                        "Torus": lambda: NoCTopology.build_torus(tile_rows, tile_cols),
                        "FoldedTorus": lambda: NoCTopology.build_folded_torus(tile_rows, tile_cols),
                        "Tree": lambda: NoCTopology.build_tree(num_tiles),
                        "Butterfly": lambda: NoCTopology.build_butterfly(num_tiles),
                        "Hypercube": lambda: NoCTopology.build_hypercube(num_tiles),
                        "Ring": lambda: NoCTopology.build_ring(num_tiles)}
        if topology not in builders:
            raise ValueError(f"Unknown topology: {topology}")
        self.topology = builders[topology]()
        self.shortest_paths, self._predecessors = {}, {}
        for src in range(num_tiles):
            dist, pred, queue = {src: 0}, {src: None}, deque([src])
            while queue:
                u = queue.popleft()
                for v in self.topology["adjacency"][u]:
                    if v not in dist:
                        dist[v], pred[v] = dist[u]+1, u
                        queue.append(v)
            if any(i not in dist for i in range(num_tiles)):
                raise ValueError("Disconnected endpoint graph")
            self.shortest_paths[src], self._predecessors[src] = dist, pred
        distances = [self.get_hop_count(s, d) for s in range(num_tiles) for d in range(num_tiles) if s != d]
        self.topology.update(diameter=max(distances, default=0), avg_hops=float(np.mean(distances)) if distances else 0)

    def get_hop_count(self, src, dst):
        return self.shortest_paths[src][dst]

    def get_route(self, src, dst):
        """Sorted-neighbor BFS route; no claim of dimension-ordered routing."""
        if not (0 <= src < self.num_tiles and 0 <= dst < self.num_tiles):
            raise ValueError("Routes must connect valid endpoints")
        pred, route, node = self._predecessors[src], [], dst
        while node != src:
            route.append((pred[node], node))
            node = pred[node]
        return route[::-1]

    def _phase_cost(self, transfers):
        links, injection, ejection = defaultdict(int), defaultdict(int), defaultdict(int)
        hops, flit_hops, payload, wire_bytes = 0, 0, 0, 0
        for src, dst, size in transfers:
            if size < 0 or int(size) != size:
                raise ValueError("Transfer payload must be a nonnegative integer")
            if not size or src == dst:
                continue
            route = self.get_route(src, dst)
            # Last partial flit occupies a whole link transfer, including chunks.
            n_flits = (int(size)+self.FLIT_SIZE_BYTES-1)//self.FLIT_SIZE_BYTES
            physical_bytes = n_flits*self.FLIT_SIZE_BYTES
            payload += size
            wire_bytes += physical_bytes
            injection[src] += physical_bytes
            ejection[dst] += physical_bytes
            hops = max(hops, len(route))
            flit_hops += n_flits*len(route)
            for edge in route:
                links[edge] += physical_bytes
        resources = [(v*8/self.bandwidth_gbps, f"link:{k[0]}->{k[1]}") for k, v in links.items()]
        resources += [(v*8/self.endpoint_bandwidth_gbps, f"injection:{k}") for k, v in injection.items()]
        resources += [(v*8/self.endpoint_bandwidth_gbps, f"ejection:{k}") for k, v in ejection.items()]
        busy = max((x[0] for x in resources), default=0)
        traversal = hops*(self.router_cycles+self.link_cycles)/self.clock_ghz
        return {"latency_ns": busy+traversal, "busy_ns": busy, "traversal_ns": traversal,
                    "link_bytes": dict(links), "bottleneck_bytes": max(links.values(), default=0),
                    "bottleneck_resources": [key for time, key in resources if math.isclose(time, busy)],
                    "flit_hops": flit_hops, "max_hops": hops, "payload_bytes": payload, "wire_endpoint_bytes": wire_bytes}

    def _schedule(self, size, protocol):
        n = self.num_tiles
        if protocol not in ("ParameterServer", "AllReduce", "RingAllReduce", "Gossip"):
            raise ValueError(f"Unknown protocol: {protocol}")
        if n == 1:
            return []
        if protocol == "ParameterServer":
            return [[(t, 0, size) for t in range(1, n)], [(0, t, size) for t in range(1, n)]]
        if protocol == "AllReduce":
            # Recursive doubling: all nodes exchange the full current partial
            # reduction in both directions at each XOR stage. Power-of-two only.
            if n & (n-1):
                raise ValueError("Recursive doubling requires power-of-two tiles")
            return [[(t, t ^ (1 << stage), size) for t in range(n)] for stage in range(n.bit_length()-1)]
        if protocol == "RingAllReduce":
            chunks = [size//n + (i < size % n) for i in range(n)]
            # Reduce-scatter sends chunk (rank-step); all-gather starts from
            # the chunk owned by that rank after n-1 reduce-scatter steps.
            return [[(t, (t+1) % n, chunks[(t-step+offset) % n]) for t in range(n)]
                    for offset in (0, 1) for step in range(n-1)]
        if protocol == "Gossip":
            # A fixed number of pairwise exchanges, NOT exact global averaging.
            rng = np.random.default_rng(42)
            phases = []
            for _ in range(math.ceil(math.log2(n))+1):
                order = rng.permutation(n).tolist()
                pairs = list(zip(order[::2], order[1::2]))
                phases.append([(s, d, size) for a, b in pairs for s, d in ((a, b), (b, a))])
            return phases
        raise ValueError(f"Unknown protocol: {protocol}")

    def simulate_fl_round(self, model_size_bytes, protocol="ParameterServer"):
        if int(model_size_bytes) != model_size_bytes or model_size_bytes < 0:
            raise ValueError("Model payload must be a nonnegative integer")
        schedule = self._schedule(int(model_size_bytes), protocol)
        phases = [self._phase_cost(x) for x in schedule]
        latency = sum(x["latency_ns"] for x in phases)
        busy = sum(x["busy_ns"] for x in phases)
        total_bytes = sum(x["payload_bytes"] for x in phases)
        flit_hops = sum(x["flit_hops"] for x in phases)
        energy = None if self.energy_profile is None else flit_hops*self.energy_profile["pj_per_flit_hop"]*1e-12
        loads = defaultdict(int)
        for phase in phases:
            for edge, count in phase["link_bytes"].items():
                loads[edge] += count
        max_load = max(loads.values(), default=0)
        mean_load = float(np.mean(list(loads.values()))) if loads else 0
        return {"protocol": protocol, "topology": self.topology_name, "total_bytes": total_bytes,
            "total_hops": sum(len(self.get_route(s, d)) for p in schedule for s, d, b in p if b),
            "flit_hops": flit_hops, "latency_ns": latency, "latency_us": latency/1000,
            "serialization_ns": busy, "traversal_ns": sum(x["traversal_ns"] for x in phases),
            "energy_j": energy, "energy_nj": None if energy is None else energy*1e9,
            "energy_status": "uncalibrated" if energy is None else "caller_supplied_calibration",
            "utilization": busy/latency if latency else 0,
            "utilization_scope": "fraction of phase estimate occupied by bottleneck serialization",
            "aggregate_throughput_gbps": total_bytes*8/latency if latency else 0,
            "bottleneck_link_bytes": max_load, "congestion_ratio": max_load/mean_load if mean_load else 1,
            "bottleneck": "server_node" if protocol == "ParameterServer" else "busiest_ring_link" if protocol == "RingAllReduce" else "phase_resource",
            "phase_bottlenecks": [p["bottleneck_resources"] for p in phases], "num_phases": len(phases),
            "exact_collective": protocol != "Gossip", "model_assumptions": self.assumptions()}

    def assumptions(self):
        return {"model": "analytical bottleneck serialization plus pipeline traversal",
            "cycle_accurate": False, "noxim_validated": False, "link_width_bits": self.link_width_bits,
            "clock_ghz": self.clock_ghz, "bandwidth_gbps": self.bandwidth_gbps,
            "rate_override": self.bandwidth_gbps != self.link_width_bits*self.clock_ghz,
            "endpoint_bandwidth_gbps": self.endpoint_bandwidth_gbps, "link_cycles": self.link_cycles,
            "router_cycles": self.router_cycles, "energy_profile": self.energy_profile,
            "technology_node": None if self.energy_profile is None else self.energy_profile["technology_node"],
            "area": None, "physical_layout_modeled": False, "reduction_compute_modeled": False,
            "server_location": "tile 0; no external controller endpoint"}

    def simulate_full_fl_training(self, model_size_bytes, num_rounds, protocol="ParameterServer"):
        if num_rounds < 0 or int(num_rounds) != num_rounds:
            raise ValueError("Round count must be a nonnegative integer")
        r = self.simulate_fl_round(model_size_bytes, protocol)
        energy = None if r["energy_nj"] is None else r["energy_nj"]*num_rounds
        return {"protocol": protocol, "topology": self.topology_name, "num_rounds": num_rounds,
            "model_size_bytes": model_size_bytes, "per_round_bytes": r["total_bytes"],
            "per_round_latency_us": r["latency_us"], "per_round_energy_nj": r["energy_nj"],
            "total_bytes": r["total_bytes"]*num_rounds, "total_latency_us": r["latency_us"]*num_rounds,
            "total_latency_ms": r["latency_us"]*num_rounds/1000, "total_energy_nj": energy,
            "total_energy_uj": None if energy is None else energy/1000,
            "total_serialization_us": r["serialization_ns"]*num_rounds/1000,
            "total_traversal_us": r["traversal_ns"]*num_rounds/1000,
            "avg_utilization": r["utilization"], "aggregate_throughput_gbps": r["aggregate_throughput_gbps"],
            "congestion_ratio": r["congestion_ratio"], "topology_diameter": self.topology["diameter"],
            "topology_bisection_bw": None, "exact_collective": r["exact_collective"],
            "phase_bottlenecks": r["phase_bottlenecks"], "model_assumptions": self.assumptions()}

    def get_topology_info(self):
        adj = self.topology["adjacency"]
        return {"name": self.topology_name, "num_tiles": self.num_tiles,
            "diameter": self.topology["diameter"], "avg_hops": self.topology["avg_hops"],
            "bisection_bandwidth": None, "num_links": sum(map(len, adj.values()))//(1 if self.topology["directed"] else 2),
            "degree": max(map(len, adj.values())), "directed": self.topology["directed"],
            "auxiliary_vertices": len(adj)-self.num_tiles, "equal_area_comparison": False}


def compare_topologies_and_protocols(num_tiles, model_size_bytes, num_rounds=20, bandwidth_gbps=None):
    results, rankings = {}, {}
    protocols = ("ParameterServer", "AllReduce", "RingAllReduce", "Gossip")
    for topology in ("Mesh", "Torus", "FoldedTorus", "Tree", "Butterfly", "Hypercube", "Ring"):
        try:
            sim = NoCSimulator(num_tiles, topology, bandwidth_gbps)
        except ValueError as exc:
            results[topology] = {p: {"error": str(exc)} for p in protocols}
            continue
        results[topology] = {"_info": sim.get_topology_info()}
        for protocol in protocols:
            try:
                results[topology][protocol] = sim.simulate_full_fl_training(model_size_bytes, num_rounds, protocol)
            except ValueError as exc:
                results[topology][protocol] = {"error": str(exc)}
    for protocol in protocols:
        latencies = {t: r[protocol]["total_latency_us"] for t, r in results.items() if "total_latency_us" in r[protocol]}
        order = sorted(latencies, key=latencies.get)
        if order:
            rankings[protocol] = {"best": order[0], "worst": order[-1], "ranking": order, "latencies": latencies,
                                      "scope": "fixed assumptions; ties are not superiority evidence",
                                      "exact_collective": protocol != "Gossip"}
    results["_rankings"] = rankings
    return results

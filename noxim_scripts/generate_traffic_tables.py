#!/usr/bin/env python3
"""
Generate Noxim traffic table files for all FL communication protocols.
Maps FL-RIS parameters to Noxim traffic format:
    src dst pir por t_on t_off t_period

Traffic table format (from Noxim GlobalTrafficTable.cpp):
  - src: source node ID (0-15 for 4x4)
  - dst: destination node ID
  - pir: packet injection rate (0.0-1.0)
  - por: probability of retransmission
  - t_on: cycle when traffic starts
  - t_off: cycle when traffic stops
  - t_period: period for repeating traffic (models FL rounds)

GNN model: 706,112 params × 4 bytes = 2,824,448 bytes
Flit: 128 bits = 16 bytes
Total flits per model: 176,528
"""

import os
import math

# --- FL-RIS Parameters (from Python experiments) ---
GRID_X = 4
GRID_Y = 4
N_TILES = GRID_X * GRID_Y  # 16
MODEL_BYTES = 706112 * 4    # 2,824,448 bytes
FLIT_BYTES = 128 // 8       # 16 bytes
FLITS_PER_MODEL = MODEL_BYTES // FLIT_BYTES  # 176,528

FL_ROUNDS = 5
RESET_TIME = 1000

# Simulation timing per FL round
CYCLES_PER_ROUND = 100000   # cycles budgeted per FL round
TOTAL_SIM = RESET_TIME + FL_ROUNDS * CYCLES_PER_ROUND

# PIR calculation: we want to inject FLITS_PER_MODEL flits over CYCLES_PER_ROUND
# PIR = flits_to_inject / cycles_available / packet_size
# With packet_size=64 flits: packets_to_inject = 176528/64 ≈ 2758 packets
# PIR = 2758 / 100000 ≈ 0.028
PACKET_SIZE = 64
PACKETS_PER_MODEL = FLITS_PER_MODEL // PACKET_SIZE
PIR_BASE = min(PACKETS_PER_MODEL / CYCLES_PER_ROUND, 0.05)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
if "noxim_scripts" in OUTPUT_DIR:
    OUTPUT_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "noxim_configs")

def node_id(x, y):
    """Convert (x,y) grid position to linear node ID."""
    return y * GRID_X + x

def node_pos(node):
    """Convert linear node ID to (x,y) grid position."""
    return (node % GRID_X, node // GRID_X)

def write_traffic(filename, entries, header_comment=""):
    """Write Noxim traffic table file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        f.write(f"% FL-RIS Traffic Table: {header_comment}\n")
        f.write(f"% Format: src dst pir por t_on t_off t_period\n")
        f.write(f"% Model: {MODEL_BYTES} bytes = {FLITS_PER_MODEL} flits\n")
        f.write(f"% FL Rounds: {FL_ROUNDS}\n")
        f.write(f"%\n")
        for entry in entries:
            f.write(" ".join(str(x) for x in entry) + "\n")
    print(f"  Written: {filepath} ({len(entries)} entries)")

def gen_ringallreduce():
    """
    RingAllReduce: nodes form a ring and each sends to its successor.
    Reduce-scatter phase: each node sends 1/N of data to next node in ring.
    Allgather phase: each node sends gathered chunks to next node.
    
    Ring order for 4x4 mesh (snake traversal):
    0→1→2→3→7→11→15→14→13→12→8→4→(back to 0)
    """
    # Snake ring through 4x4 grid
    ring = []
    for y in range(GRID_Y):
        if y % 2 == 0:
            ring.extend([node_id(x, y) for x in range(GRID_X)])
        else:
            ring.extend([node_id(x, y) for x in range(GRID_X-1, -1, -1)])
    
    # Flits per segment in RingAllReduce: model_size / N_tiles
    flits_per_segment = FLITS_PER_MODEL // N_TILES
    pir_segment = min(flits_per_segment / PACKET_SIZE / CYCLES_PER_ROUND, 0.05)
    
    entries = []
    for round_idx in range(FL_ROUNDS):
        t_on = RESET_TIME + round_idx * CYCLES_PER_ROUND
        t_off = t_on + CYCLES_PER_ROUND - 1
        t_period = TOTAL_SIM + 1  # no repeat
        
        for i in range(len(ring)):
            src = ring[i]
            dst = ring[(i + 1) % len(ring)]
            entries.append([src, dst, f"{pir_segment:.6f}", f"{pir_segment:.6f}", t_on, t_off, t_period])
    
    write_traffic("traffic_ringallreduce.txt", entries,
                  f"RingAllReduce ({N_TILES} nodes, {FL_ROUNDS} rounds)")

def gen_paramserver():
    """
    ParameterServer: All tiles send to server (node 0), server broadcasts back.
    Phase 1 (upload):   nodes 1-15 → node 0
    Phase 2 (broadcast): node 0 → nodes 1-15
    """
    pir = min(PACKETS_PER_MODEL / CYCLES_PER_ROUND, 0.05)
    half_round = CYCLES_PER_ROUND // 2
    
    entries = []
    for round_idx in range(FL_ROUNDS):
        t_base = RESET_TIME + round_idx * CYCLES_PER_ROUND
        
        # Upload phase (first half of round)
        for node in range(1, N_TILES):
            t_on = t_base
            t_off = t_base + half_round - 1
            entries.append([node, 0, f"{pir:.6f}", f"{pir:.6f}", t_on, t_off, TOTAL_SIM + 1])
        
        # Broadcast phase (second half of round)
        for node in range(1, N_TILES):
            t_on = t_base + half_round
            t_off = t_base + CYCLES_PER_ROUND - 1
            entries.append([0, node, f"{pir:.6f}", f"{pir:.6f}", t_on, t_off, TOTAL_SIM + 1])
    
    write_traffic("traffic_paramserver.txt", entries,
                  f"ParameterServer (server=node0, {FL_ROUNDS} rounds)")

def gen_allreduce():
    """
    AllReduce (butterfly): pairwise exchanges in log2(N) stages.
    Stage k: node i exchanges with node i XOR (1 << k).
    """
    stages = int(math.log2(N_TILES))
    stage_cycles = CYCLES_PER_ROUND // stages
    flits_per_stage = FLITS_PER_MODEL // stages
    pir = min(flits_per_stage / PACKET_SIZE / stage_cycles, 0.05)
    
    entries = []
    for round_idx in range(FL_ROUNDS):
        t_base = RESET_TIME + round_idx * CYCLES_PER_ROUND
        
        for stage in range(stages):
            t_on = t_base + stage * stage_cycles
            t_off = t_on + stage_cycles - 1
            
            for node in range(N_TILES):
                partner = node ^ (1 << stage)
                if partner < N_TILES and partner != node:
                    entries.append([node, partner, f"{pir:.6f}", f"{pir:.6f}", t_on, t_off, TOTAL_SIM + 1])
    
    write_traffic("traffic_allreduce.txt", entries,
                  f"AllReduce butterfly ({stages} stages, {FL_ROUNDS} rounds)")

def gen_gossip():
    """
    Gossip: Each node exchanges with a random neighbor each round.
    Using deterministic neighbor pattern for reproducibility.
    """
    pir = min(PACKETS_PER_MODEL / CYCLES_PER_ROUND / 2, 0.05)
    
    entries = []
    for round_idx in range(FL_ROUNDS):
        t_on = RESET_TIME + round_idx * CYCLES_PER_ROUND
        t_off = t_on + CYCLES_PER_ROUND - 1
        
        for node in range(N_TILES):
            x, y = node_pos(node)
            # Each node exchanges with right and down neighbors
            neighbors = []
            if x < GRID_X - 1:
                neighbors.append(node_id(x+1, y))
            if y < GRID_Y - 1:
                neighbors.append(node_id(x, y+1))
            
            for nbr in neighbors:
                entries.append([node, nbr, f"{pir:.6f}", f"{pir:.6f}", t_on, t_off, TOTAL_SIM + 1])
                entries.append([nbr, node, f"{pir:.6f}", f"{pir:.6f}", t_on, t_off, TOTAL_SIM + 1])
    
    write_traffic("traffic_gossip.txt", entries,
                  f"Gossip (neighbor exchange, {FL_ROUNDS} rounds)")

def gen_dutycycle():
    """
    Duty cycling: Only 4 of 16 tiles active (Threshold -10dB = 25% active).
    Active tiles: 0, 5, 10, 15 (diagonal — best coverage).
    Sleeping tiles: all others (PIR=0 effectively by absence from table).
    Uses RingAllReduce among active tiles only.
    """
    active_tiles = [0, 5, 10, 15]
    flits_per_segment = FLITS_PER_MODEL // len(active_tiles)
    pir = min(flits_per_segment / PACKET_SIZE / CYCLES_PER_ROUND, 0.05)
    
    entries = []
    for round_idx in range(FL_ROUNDS):
        t_on = RESET_TIME + round_idx * CYCLES_PER_ROUND
        t_off = t_on + CYCLES_PER_ROUND - 1
        
        for i in range(len(active_tiles)):
            src = active_tiles[i]
            dst = active_tiles[(i + 1) % len(active_tiles)]
            entries.append([src, dst, f"{pir:.6f}", f"{pir:.6f}", t_on, t_off, TOTAL_SIM + 1])
    
    write_traffic("traffic_dutycycle.txt", entries,
                  f"Duty Cycling (4/16 active, RingAllReduce, {FL_ROUNDS} rounds)")

def gen_noniid_skewed():
    """
    Non-IID (alpha=0.1): Highly skewed injection rates.
    Tiles with more data inject at higher PIR.
    Dirichlet allocation: some tiles have 3-5x the data of others.
    Uses ParameterServer pattern with varying PIR per tile.
    """
    # Simulate Dirichlet alpha=0.1: highly skewed
    # Relative data amounts (normalized)
    skew_factors = [3.0, 0.2, 0.1, 2.5, 0.3, 4.0, 0.1, 0.5,
                    1.0, 0.1, 0.2, 3.5, 0.4, 0.1, 2.0, 0.1]
    total_skew = sum(skew_factors)
    
    half_round = CYCLES_PER_ROUND // 2
    
    entries = []
    for round_idx in range(FL_ROUNDS):
        t_base = RESET_TIME + round_idx * CYCLES_PER_ROUND
        
        for node in range(1, N_TILES):
            # PIR proportional to data amount
            node_pir = min(PIR_BASE * skew_factors[node] / (total_skew / N_TILES), 0.05)
            t_on = t_base
            t_off = t_base + half_round - 1
            entries.append([node, 0, f"{node_pir:.6f}", f"{node_pir:.6f}", t_on, t_off, TOTAL_SIM + 1])
        
        # Broadcast from server (uniform)
        for node in range(1, N_TILES):
            t_on = t_base + half_round
            t_off = t_base + CYCLES_PER_ROUND - 1
            entries.append([0, node, f"{PIR_BASE:.6f}", f"{PIR_BASE:.6f}", t_on, t_off, TOTAL_SIM + 1])
    
    write_traffic("traffic_noniid_alpha01.txt", entries,
                  f"Non-IID alpha=0.1 (skewed PIR, ParameterServer, {FL_ROUNDS} rounds)")

def gen_compression_int8():
    """
    INT8 compression: model size is 4x smaller.
    706,112 params × 1 byte = 706,112 bytes = 44,132 flits.
    """
    compressed_flits = MODEL_BYTES // 4 // FLIT_BYTES  # INT8 = 1/4 of FP32
    flits_per_segment = compressed_flits // N_TILES
    pir = min(flits_per_segment / PACKET_SIZE / CYCLES_PER_ROUND, 0.05)
    
    # Same RingAllReduce pattern but with lower PIR (less data)
    ring = []
    for y in range(GRID_Y):
        if y % 2 == 0:
            ring.extend([node_id(x, y) for x in range(GRID_X)])
        else:
            ring.extend([node_id(x, y) for x in range(GRID_X-1, -1, -1)])
    
    entries = []
    for round_idx in range(FL_ROUNDS):
        t_on = RESET_TIME + round_idx * CYCLES_PER_ROUND
        t_off = t_on + CYCLES_PER_ROUND - 1
        
        for i in range(len(ring)):
            src = ring[i]
            dst = ring[(i + 1) % len(ring)]
            entries.append([src, dst, f"{pir:.6f}", f"{pir:.6f}", t_on, t_off, TOTAL_SIM + 1])
    
    write_traffic("traffic_ringallreduce_int8.txt", entries,
                  f"RingAllReduce INT8 ({compressed_flits} flits/model, {FL_ROUNDS} rounds)")

if __name__ == "__main__":
    print("Generating Noxim FL-RIS traffic tables...")
    print(f"  Model: {MODEL_BYTES:,} bytes = {FLITS_PER_MODEL:,} flits")
    print(f"  Packets per model (size={PACKET_SIZE}): {PACKETS_PER_MODEL:,}")
    print(f"  Base PIR: {PIR_BASE:.6f}")
    print()
    
    gen_ringallreduce()
    gen_paramserver()
    gen_allreduce()
    gen_gossip()
    gen_dutycycle()
    gen_noniid_skewed()
    gen_compression_int8()
    
    print("\nAll traffic tables generated!")

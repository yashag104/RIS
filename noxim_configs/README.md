# FL-RIS Noxim Integration — Step-by-Step Guide

## Overview
This directory contains everything needed to run the FL-RIS NoC simulations
on Noxim, using optimal parameters from the Python experiment suite.

## Directory Structure
```
noxim_configs/          ← Noxim YAML configs and traffic tables
  fl_ris_mesh_4x4.yaml       Base 4x4 Mesh config
  fl_ris_torus_4x4.yaml      4x4 Torus config (needs patch)
  traffic_ringallreduce.txt   RingAllReduce FL traffic pattern
  traffic_paramserver.txt     ParameterServer FL traffic
  traffic_allreduce.txt       AllReduce (butterfly) traffic
  traffic_gossip.txt          Gossip neighbor exchange traffic
  traffic_dutycycle.txt       Duty cycling (4/16 active tiles)
  traffic_noniid_alpha01.txt  Non-IID skewed injection rates
  traffic_ringallreduce_int8.txt  INT8 compressed model traffic

noxim_patch/            ← Torus topology modification
  torus_topology_patch.cpp    All code + instructions for Torus support

noxim_scripts/          ← Automation
  generate_traffic_tables.py  Regenerate traffic tables
  run_all_noxim.sh            Master experiment runner (8 experiments)
  parse_noxim_output.py       Parse Noxim output → JSON/CSV
  compare_python_noxim.py     Python vs Noxim comparison plots
```

## Quick Start (Mesh — No Patch Needed)

```bash
# 1. Copy files to your Ubuntu Noxim machine
scp -r noxim_configs noxim_scripts user@ubuntu:~/fl_ris_noxim/

# 2. SSH to Ubuntu
ssh user@ubuntu

# 3. Run a single experiment
cd ~/fl_ris_noxim
~/noxim/bin/noxim -config noxim_configs/fl_ris_mesh_4x4.yaml

# 4. Run ALL experiments
chmod +x noxim_scripts/run_all_noxim.sh
./noxim_scripts/run_all_noxim.sh ~/noxim/bin/noxim

# 5. Parse results
python3 noxim_scripts/parse_noxim_output.py results/noxim_results/

# 6. Generate comparison plots
python3 noxim_scripts/compare_python_noxim.py results/noxim_results/
```

## Torus Topology Setup (Full FL-RIS Optimal Config)

### Step 1: Apply Torus Patch to Noxim Source

Open `noxim_patch/torus_topology_patch.cpp` and follow the 10 steps:

```bash
cd ~/noxim

# Step 1: Add TOPOLOGY_TORUS constant
# In src/GlobalParams.h, add after #define TOPOLOGY_MESH "MESH":
#   #define TOPOLOGY_TORUS "TORUS"

# Step 2: Add buildTorus() declaration
# In src/NoC.h, add in private section:
#   void buildTorus();

# Step 3: Add constructor case
# In src/NoC.h SC_CTOR, add:
#   else if (GlobalParams::topology == TOPOLOGY_TORUS) buildTorus();

# Step 4: Add buildTorus() implementation
# In src/NoC.cpp, paste the buildTorus() function from the patch file

# Step 5-6: Create Routing_TORUS_XY.h and .cpp
# Copy from patch file into src/routingAlgorithms/

# Step 7-9: Register the routing algorithm

# Step 10: Update Makefile to include new source files
```

### Step 2: Rebuild Noxim
```bash
cd ~/noxim/bin
make clean
make
```

### Step 3: Verify Torus
```bash
# Test with uniform random traffic
./noxim -dimx 4 -dimy 4 -topology TORUS -routing TORUS_XY -traffic random -sim 10000

# Compare latency: Torus should be ~25% lower than Mesh
./noxim -dimx 4 -dimy 4 -traffic random -sim 10000  # Mesh baseline
```

### Step 4: Run Full Torus Experiments
```bash
# The run_all_noxim.sh script auto-detects fl_ris_torus_4x4.yaml
./noxim_scripts/run_all_noxim.sh ~/noxim/bin/noxim
```

## Parameters from Python Experiments

| Parameter | Optimal Value | Source Experiment |
|---|---|---|
| Topology | Torus 4×4 | Exp 14 |
| Protocol | RingAllReduce | Exp 15 |
| FL Algorithm | FedAvg (5 rounds) | Exp 11 |
| Model | GNN (706K params = 2.76 MB) | Exp 12 |
| Compression | INT8 (4× less comm) | Exp 3 |
| Duty Cycling | Threshold -10dB (25% active) | Exp 18 |
| Non-IID | α=0.5 (fairness=0.70) | Exp 5 |
| Local Epochs | 5 | Exp 1 |
| Buffer | 8 flits | Sweep F |

## Experiment Descriptions

| ID | Experiment | What It Tests |
|---|---|---|
| A | Topology | Mesh vs Torus (latency, energy) |
| B | Protocol | RingAllReduce vs PS vs AllReduce vs Gossip |
| C | Compression | FP32 vs INT8 packet sizes |
| D | Duty Cycling | All 16 vs 4 active tiles |
| E | Non-IID | Uniform vs skewed injection rates |
| F | Buffer Depth | 4, 8, 16, 32 flit buffers |
| G | Scalability | 2×2 to 4×4 grid sizes |
| H | Load Analysis | PIR sweep from 0.001 to 0.1 |

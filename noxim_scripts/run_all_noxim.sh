#!/bin/bash
# ============================================================================
# FL-RIS Noxim Experiment Suite
# Run all NoC experiments with optimal parameters from Python simulation
# ============================================================================
#
# Usage: ./run_all_noxim.sh [NOXIM_PATH]
#   NOXIM_PATH: path to noxim binary (default: ~/noxim/bin/noxim)
#
# Prerequisites:
#   1. Noxim installed and compiled (with Torus patch if using Torus)
#   2. Traffic tables generated: python3 generate_traffic_tables.py
#   3. Config files in noxim_configs/
#
# ============================================================================

set -e

NOXIM="${1:-$HOME/noxim/bin/noxim}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../noxim_configs"
RESULTS_DIR="$SCRIPT_DIR/../results/noxim_results"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " FL-RIS Noxim Experiment Suite"
echo "============================================"
echo "  Noxim: $NOXIM"
echo "  Configs: $CONFIG_DIR"
echo "  Results: $RESULTS_DIR"
echo ""

# Check noxim exists
if [ ! -f "$NOXIM" ]; then
    echo "[ERROR] Noxim not found at $NOXIM"
    echo "  Please provide path: ./run_all_noxim.sh /path/to/noxim"
    exit 1
fi

run_experiment() {
    local name="$1"
    local config="$2"
    local traffic="$3"
    local extra_args="${4:-}"
    
    echo "──────────────────────────────────────────"
    echo "  Experiment: $name"
    echo "  Config: $config"
    echo "  Traffic: $traffic"
    echo "──────────────────────────────────────────"
    
    local output_file="$RESULTS_DIR/${name}.txt"
    local start_time=$(date +%s)
    
    # Build command
    local cmd="$NOXIM -config $CONFIG_DIR/$config"
    if [ -n "$traffic" ]; then
        cmd="$cmd -traffic table -table $CONFIG_DIR/$traffic"
    fi
    if [ -n "$extra_args" ]; then
        cmd="$cmd $extra_args"
    fi
    
    echo "  CMD: $cmd"
    eval "$cmd" > "$output_file" 2>&1
    
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo "  ✅ Done in ${elapsed}s → $output_file"
    echo ""
}

# ============================================================================
# EXPERIMENT A: Topology Comparison (Mesh vs Torus)
# ============================================================================
echo ""
echo "═══════════════════════════════════════════"
echo " EXPERIMENT A: Topology Comparison"
echo "═══════════════════════════════════════════"

# A1: Mesh + RingAllReduce (baseline)
run_experiment "A1_mesh_ringallreduce" \
    "fl_ris_mesh_4x4.yaml" \
    "traffic_ringallreduce.txt"

# A2: Mesh + ParameterServer
run_experiment "A2_mesh_paramserver" \
    "fl_ris_mesh_4x4.yaml" \
    "traffic_paramserver.txt"

# If Torus config exists (after applying patch)
if [ -f "$CONFIG_DIR/fl_ris_torus_4x4.yaml" ]; then
    run_experiment "A3_torus_ringallreduce" \
        "fl_ris_torus_4x4.yaml" \
        "traffic_ringallreduce.txt"
    
    run_experiment "A4_torus_paramserver" \
        "fl_ris_torus_4x4.yaml" \
        "traffic_paramserver.txt"
fi

# ============================================================================
# EXPERIMENT B: Protocol Comparison (on Mesh)
# ============================================================================
echo ""
echo "═══════════════════════════════════════════"
echo " EXPERIMENT B: Protocol Comparison"
echo "═══════════════════════════════════════════"

run_experiment "B1_ringallreduce" \
    "fl_ris_mesh_4x4.yaml" \
    "traffic_ringallreduce.txt"

run_experiment "B2_paramserver" \
    "fl_ris_mesh_4x4.yaml" \
    "traffic_paramserver.txt"

run_experiment "B3_allreduce" \
    "fl_ris_mesh_4x4.yaml" \
    "traffic_allreduce.txt"

run_experiment "B4_gossip" \
    "fl_ris_mesh_4x4.yaml" \
    "traffic_gossip.txt"

# ============================================================================
# EXPERIMENT C: Compression Impact (FP32 vs INT8)
# ============================================================================
echo ""
echo "═══════════════════════════════════════════"
echo " EXPERIMENT C: Compression Impact"
echo "═══════════════════════════════════════════"

# C1: FP32 (full model, same as B1)
run_experiment "C1_fp32_ringallreduce" \
    "fl_ris_mesh_4x4.yaml" \
    "traffic_ringallreduce.txt"

# C2: INT8 (4x smaller)
run_experiment "C2_int8_ringallreduce" \
    "fl_ris_mesh_4x4.yaml" \
    "traffic_ringallreduce_int8.txt"

# ============================================================================
# EXPERIMENT D: Duty Cycling (25% active tiles)
# ============================================================================
echo ""
echo "═══════════════════════════════════════════"
echo " EXPERIMENT D: Duty Cycling"
echo "═══════════════════════════════════════════"

# D1: All 16 tiles active (baseline = B1)
run_experiment "D1_all_active" \
    "fl_ris_mesh_4x4.yaml" \
    "traffic_ringallreduce.txt"

# D2: 4/16 tiles active (Threshold -10dB)
run_experiment "D2_dutycycle_25pct" \
    "fl_ris_mesh_4x4.yaml" \
    "traffic_dutycycle.txt"

# ============================================================================
# EXPERIMENT E: Non-IID Congestion
# ============================================================================
echo ""
echo "═══════════════════════════════════════════"
echo " EXPERIMENT E: Non-IID Traffic"
echo "═══════════════════════════════════════════"

# E1: Uniform (IID) = B2 (ParameterServer)
run_experiment "E1_iid_paramserver" \
    "fl_ris_mesh_4x4.yaml" \
    "traffic_paramserver.txt"

# E2: Skewed (Non-IID alpha=0.1)
run_experiment "E2_noniid_alpha01" \
    "fl_ris_mesh_4x4.yaml" \
    "traffic_noniid_alpha01.txt"

# ============================================================================
# EXPERIMENT F: Buffer Depth Sweep
# ============================================================================
echo ""
echo "═══════════════════════════════════════════"
echo " EXPERIMENT F: Buffer Depth Sweep"
echo "═══════════════════════════════════════════"

for BDEPTH in 4 8 16 32; do
    run_experiment "F_buffer_${BDEPTH}" \
        "fl_ris_mesh_4x4.yaml" \
        "traffic_ringallreduce.txt" \
        "-buffer $BDEPTH"
done

# ============================================================================
# EXPERIMENT G: Scalability (different grid sizes)
# ============================================================================
echo ""
echo "═══════════════════════════════════════════"
echo " EXPERIMENT G: Scalability"
echo "═══════════════════════════════════════════"

for DIMX in 2 3 4; do
    for DIMY in 2 3 4; do
        TILES=$((DIMX * DIMY))
        if [ $TILES -le 16 ]; then
            run_experiment "G_${DIMX}x${DIMY}_mesh" \
                "fl_ris_mesh_4x4.yaml" \
                "" \
                "-dimx $DIMX -dimy $DIMY -traffic random"
        fi
    done
done

# ============================================================================
# EXPERIMENT H: Injection Rate Sweep (load analysis)
# ============================================================================
echo ""
echo "═══════════════════════════════════════════"
echo " EXPERIMENT H: Injection Rate Sweep"
echo "═══════════════════════════════════════════"

for PIR in 0.001 0.005 0.01 0.02 0.05 0.1; do
    run_experiment "H_pir_${PIR}" \
        "fl_ris_mesh_4x4.yaml" \
        "" \
        "-pir $PIR -traffic random"
done

# ============================================================================
echo ""
echo "============================================"
echo " ALL EXPERIMENTS COMPLETE"
echo " Results saved to: $RESULTS_DIR"
echo "============================================"
echo ""
echo "Next: Run parse_noxim_output.py to extract metrics"
echo "  python3 $SCRIPT_DIR/parse_noxim_output.py $RESULTS_DIR"

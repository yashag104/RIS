"""
Configuration for Federated Learning on Distributed RIS Tiles
==============================================================
Best-performing settings validated by experiments 11-20 and literature:

Architecture: GNN/GAT        [Shen et al., IEEE TSP 2021; He et al., IEEE TWC 2022]
Optimizer:    ADMM (baseline) [Yu et al., IEEE JSAC 2020; Huang et al., IEEE TWC 2019]
Topology:     Torus           [Dally & Towles, Morgan Kaufmann 2004]
Protocol:     RingAllReduce   [Sergeev & Del Balso, 2018; Patarasuk & Yuan, JPDC 2009]
FL Algo:      FedAvg          [McMahan et al., AISTATS 2017]
Duty Cycling: Threshold -10dB [Exp 18: 70% energy savings, <0.01 dB SNR loss]
Quantization: 3-bit (>95%)    [Wu & Zhang, IEEE TWC 2020]
"""

import torch


class Config:
    # ============ System Parameters ============
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    RANDOM_SEEDS = [42, 123, 456, 789, 1024]  # For multi-seed experiments

    # ============ Tile-Pixel Grid Configuration ============
    TILE_GRID_ROWS = 4  # Tiles in row direction
    TILE_GRID_COLS = 4  # Tiles in column direction
    PIXEL_GRID_ROWS = 8  # Pixels per tile (rows)
    PIXEL_GRID_COLS = 8  # Pixels per tile (cols)
    
    # Derived RIS Parameters
    NUM_TILES = TILE_GRID_ROWS * TILE_GRID_COLS  # 16 tiles (4x4)
    ELEMENTS_PER_TILE = PIXEL_GRID_ROWS * PIXEL_GRID_COLS  # 64 pixels (8x8)
    TOTAL_RIS_ELEMENTS = NUM_TILES * ELEMENTS_PER_TILE  # 1024 elements

    # ============ Environment Parameters ============
    FREQUENCY = 28e9  # Operating frequency (28 GHz mmWave)
    WAVELENGTH = 3e8 / FREQUENCY  # Wavelength (~10.7mm)
    ROOM_SIZE = (10, 10, 3)  # Room dimensions (x, y, z) in meters
    CHIP_AREA_M2 = ROOM_SIZE[0] * ROOM_SIZE[1]  # Chip/room area in m²
    NUM_USERS = 10  # Number of users in the environment

    # ============ Channel Model Parameters ============
    PATH_LOSS_EXPONENT = 2.5
    NOISE_POWER_DBM = -90  # Noise power in dBm
    TX_POWER_DBM = 30  # Transmit power in dBm
    
    # Rician Fading
    RICIAN_K_FACTOR_DB = 10.0  # K-factor in dB (LoS component strength)
    CHANNEL_SCENARIO = "LoS"  # "LoS", "NLoS", or "mixed"
    
    # Spatial Correlation
    SPATIAL_CORRELATION_RHO = 0.7  # Adjacent element correlation (0-1)
    
    # Multi-path
    NUM_PATHS = 5  # Number of NLoS multipath components
    
    # CSI Estimation Error
    CSI_ERROR_VARIANCE = 0.0  # 0 = perfect CSI, >0 adds estimation noise
    
    # Phase Noise (hardware imperfection)
    PHASE_NOISE_STD_DEG = 0.0  # Phase noise std in degrees (0=ideal, 2/5/10=realistic)
    
    # Phase Quantization
    PHASE_QUANTIZATION_BITS = 0  # 0=continuous, 1/2/3-bit for discrete

    # ============ DeepMIMO Dataset Configuration ============
    USE_DEEPMIMO = False  # O1_28 data not available; using synthetic Rician channel
    DEEPMIMO_SCENARIO = "O1_28"  # Outdoor 28 GHz urban
    DEEPMIMO_DATA_DIR = ".."  # O1_28/ scenario folder is in SOP-main/ (parent of Codebase/)
    
    # ============ Multi-Scenario Dataset Configuration ============
    DATASET_SCENARIOS = {
        'deepmimo_O1_28': {'type': 'deepmimo', 'scenario': 'O1_28', 'desc': 'Outdoor 28 GHz'},
        'deepmimo_O1_60': {'type': 'deepmimo', 'scenario': 'O1_60', 'desc': 'Outdoor 60 GHz'},
        'deepmimo_I3_60': {'type': 'deepmimo', 'scenario': 'I3_60', 'desc': 'Indoor 60 GHz'},
        '3gpp_umi_28':    {'type': '3gpp_umi', 'frequency': 28e9, 'desc': '3GPP UMi 28 GHz'},
        'synthetic_rician':{'type': 'synthetic', 'desc': 'Synthetic Rician (fallback)'},
    }
    ACTIVE_SCENARIOS = ['deepmimo_O1_28', '3gpp_umi_28', 'synthetic_rician']
    
    # ============ Channel Model Type ============
    # Options: 'rician', '3gpp_umi'
    CHANNEL_MODEL_TYPE = 'rician'
    
    # ============ DRL Hyperparameters ============
    DRL_ACTOR_LR = 3e-4  # Learning rate for actor
    DRL_CRITIC_LR = 3e-4 # Learning rate for critic
    DRL_BATCH_SIZE = 256
    DRL_DISCOUNT = 0.99
    DRL_TAU = 0.005
    DRL_POLICY_NOISE = 0.2
    DRL_NOISE_CLIP = 0.5
    DRL_POLICY_FREQ = 2

    # ============ Federated Learning Parameters ============
    FL_ROUNDS = 20  # Total federated learning rounds
    LOCAL_EPOCHS = 3  # Local training epochs per round
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    GNN_LEARNING_RATE = 0.0003  # Lower LR for GNN (GAT is sensitive to high LR)
    
    # FedProx
    FEDPROX_MU = 0.01  # Proximal term coefficient (0 = FedAvg)

    # ============ Model Architecture ============
    # GNN (GAT) selected: exploits inter-element spatial correlations via attention
    # [Shen et al., IEEE TSP 2021; He et al., IEEE TWC 2022]
    MODEL_TYPE = "GNN"
    
    # MLP Parameters
    HIDDEN_DIM = 256
    NUM_LAYERS = 3
    DROPOUT = 0.1
    
    # GNN Parameters
    GNN_HIDDEN_DIM = 256    # Was 32 — scaled up for 64-element phase prediction capacity
    GNN_NUM_LAYERS = 3      # Was 2 — deeper GAT for multi-hop message passing
    GNN_NUM_HEADS = 8       # Was 4 — more attention heads for richer representations
    
    # CNN+Attention Parameters
    CNN_HIDDEN_CHANNELS = 64
    CNN_USE_SE_ATTENTION = True  # Squeeze-and-Excitation block
    CNN_SE_REDUCTION = 16  # SE reduction ratio
    
    # Transformer Parameters
    TRANSFORMER_D_MODEL = 256
    TRANSFORMER_NUM_HEADS = 8
    TRANSFORMER_NUM_LAYERS = 4
    TRANSFORMER_FF_DIM = 512

    # ============ Dataset Parameters ============
    TRAIN_SAMPLES = 2000  # Training samples per tile (scaled up from 500)
    TEST_SAMPLES = 2000  # Test samples (held-out region)
    NON_IID_ALPHA = 0.5  # Dirichlet parameter for non-IID data (lower = more non-IID)

    # ============ Communication Parameters ============
    COMMUNICATION_ROUNDS_LOG = []  # Track communication per round
    PACKET_SIZE_BYTES = 4  # Size of float32 in bytes (local computation)
    COMM_BYTES_PER_PARAM = 1  # INT8 quantized transmission (1 byte/param)
    NOC_BANDWIDTH_GBPS = 10  # Network-on-Chip bandwidth
    TARGET_NOC_UTILIZATION = 0.8  # Target max utilization (<80%)

    # ============ NoC Topology Configuration ============
    # Torus: ~33% fewer hops than Mesh via wrap-around links [Dally & Towles, 2004]
    NOC_TOPOLOGY = "Torus"
    # RingAllReduce: bandwidth-optimal aggregation [Patarasuk & Yuan, JPDC 2009]
    NOC_PROTOCOL = "RingAllReduce"
    
    # ============ Sleep Scheduling Parameters ============
    SLEEP_SCHEDULING_ENABLED = True  # Enable dynamic sleep scheduling
    SLEEP_SIGNAL_THRESHOLD = 0.1  # Signal strength threshold to wake (normalized)
    SLEEP_CHECK_INTERVAL = 5  # Check wake condition every N rounds
    ACTIVE_POWER_TILE = 1.0  # Active power per tile (W)
    SLEEP_POWER_TILE = 0.05  # Sleep power per tile (W) - much lower than idle

    # ============ Energy Parameters ============
    ENERGY_PER_FLOP = 1e-12  # Energy per floating point operation (J)
    ENERGY_PER_BIT = 1e-9  # Energy per bit transmitted (J)
    IDLE_POWER_TILE = 0.1  # Idle power per tile (W)

    # ============ Optimization Weights (for composite score) ============
    WEIGHT_SNR = 0.4  # Weight for SNR optimization
    WEIGHT_ENERGY = 0.3  # Weight for energy optimization
    WEIGHT_COMM = 0.3  # Weight for communication optimization

    # ============ Aggregation Method ============
    # FedAvg: efficient with increased local computation [McMahan et al., AISTATS 2017]
    # Switch to FedProx (mu=0.01) for highly non-IID settings [Li et al., MLSys 2020]
    AGGREGATION_METHOD = "FedAvg"

    # ============ Save Paths ============
    RESULTS_DIR = "results/"
    MODELS_DIR = "models/saved/"
    PLOTS_DIR = "plots/"
    METRICS_DIR = "metrics/"
    DATA_DIR = "data/"

    # ============ Logging ============
    VERBOSE = True
    SAVE_EVERY_N_ROUNDS = 10
    
    # ============ Dynamic Duty Cycling (Pixel-Level) ============
    DUTY_CYCLE_ENABLED = True  # Validated by Exp 18: 70% energy savings, <0.01 dB SNR loss
    DUTY_CYCLE_THRESHOLD_DB = -10  # CSI power threshold to turn pixel ON
    DUTY_CYCLE_MIN_ACTIVE_RATIO = 0.25  # At least 25% pixels always ON
    DUTY_CYCLE_STRATEGY = 'threshold'  # 'threshold', 'topk', 'adaptive'
    ACTIVE_POWER_PIXEL = 0.015  # Active power per pixel (W)
    SLEEP_POWER_PIXEL = 0.001  # Sleep power per pixel (W)
    
    # ============ Sweep Ranges for Systematic Experiments ============
    CHIP_AREAS_M2 = [25, 50, 100, 200, 400]
    TILE_COUNTS = [4, 9, 16, 25, 36, 49, 64]
    PIXEL_COUNTS = [16, 36, 64, 100, 144, 196, 256]

    @classmethod
    def get_config_dict(cls):
        """Returns configuration as dictionary for saving"""
        return {k: v for k, v in cls.__dict__.items()
                if not k.startswith('_') and not callable(v)}
    
    @classmethod
    def update_tile_config(cls, tile_rows, tile_cols, pixel_rows=8, pixel_cols=8):
        """Update tile and pixel configuration dynamically"""
        cls.TILE_GRID_ROWS = tile_rows
        cls.TILE_GRID_COLS = tile_cols
        cls.PIXEL_GRID_ROWS = pixel_rows
        cls.PIXEL_GRID_COLS = pixel_cols
        cls.NUM_TILES = tile_rows * tile_cols
        cls.ELEMENTS_PER_TILE = pixel_rows * pixel_cols
        cls.TOTAL_RIS_ELEMENTS = cls.NUM_TILES * cls.ELEMENTS_PER_TILE
    
    @classmethod
    def update_room_size(cls, x, y, z=3):
        """Update room/chip dimensions"""
        cls.ROOM_SIZE = (x, y, z)
        cls.CHIP_AREA_M2 = x * y
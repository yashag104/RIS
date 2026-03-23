"""Quick smoke test for Phase 1-3 changes"""
import sys, os, numpy as np, torch
sys.path.insert(0, '.')
from config import Config
from src.channel_model import RicianChannel, generate_ris_channel_dataset
from src.dataset_utils import RISChannelDataset, create_non_iid_datasets, create_test_dataset
from models.ris_net import RISNet, create_model
from src.client import RISClient
from src.server import FederatedServer

# Use smaller scale for smoke test
Config.TRAIN_SAMPLES = 100
Config.TEST_SAMPLES = 100
Config.NUM_TILES = 4
Config.TILE_GRID_ROWS = 2
Config.TILE_GRID_COLS = 2
Config.NUM_USERS = 4
Config.FL_ROUNDS = 2
Config.LOCAL_EPOCHS = 1
Config.USE_DEEPMIMO = False
Config.VERBOSE = False
Config.BATCH_SIZE = 32
np.random.seed(42)
torch.manual_seed(42)

print("=== Generating Non-IID Datasets ===")
datasets, tile_pos = create_non_iid_datasets(Config, Config.NUM_TILES)
print(f"Created {len(datasets)} datasets, sizes: {[len(d) for d in datasets]}")
print(f"Feature dim: {datasets[0].get_input_dim()}, Label dim: {datasets[0].labels.shape[1]}")

print("=== Generating Test Dataset ===")
test_ds = create_test_dataset(Config)
print(f"Test: {len(test_ds)} samples, feature_dim={test_ds.get_input_dim()}")

print("=== Creating Models ===")
input_dim = datasets[0].get_input_dim()
num_elements = Config.ELEMENTS_PER_TILE

for mt in ["MLP", "CNN_Attention", "GNN"]:
    model = create_model(mt, input_dim, num_elements, config=Config)
    x = torch.randn(4, input_dim)
    y = model(x)
    print(f"  {mt}: params={model.count_parameters()}, output={y.shape}, range=[{y.min():.2f}, {y.max():.2f}]")

print("=== FL Training (FedAvg) ===")
Config.AGGREGATION_METHOD = "FedAvg"
global_model = create_model("MLP", input_dim, num_elements, config=Config)
server = FederatedServer(global_model, Config)
clients = [RISClient(i, create_model("MLP", input_dim, num_elements, config=Config), datasets[i], Config) for i in range(Config.NUM_TILES)]
for r in range(Config.FL_ROUNDS):
    metrics = server.aggregate_round(clients, r)
print(f"  FedAvg final loss: {metrics['avg_client_loss']:.6f}")

print("=== FL Training (FedProx) ===")
Config.AGGREGATION_METHOD = "FedProx"
global_model2 = create_model("MLP", input_dim, num_elements, config=Config)
server2 = FederatedServer(global_model2, Config)
clients2 = [RISClient(i, create_model("MLP", input_dim, num_elements, config=Config), datasets[i], Config) for i in range(Config.NUM_TILES)]
for r in range(Config.FL_ROUNDS):
    metrics2 = server2.aggregate_round(clients2, r)
print(f"  FedProx final loss: {metrics2['avg_client_loss']:.6f}")

print("=== FL Training (SCAFFOLD) ===")
Config.AGGREGATION_METHOD = "SCAFFOLD"
global_model3 = create_model("MLP", input_dim, num_elements, config=Config)
server3 = FederatedServer(global_model3, Config)
clients3 = [RISClient(i, create_model("MLP", input_dim, num_elements, config=Config), datasets[i], Config) for i in range(Config.NUM_TILES)]
for r in range(Config.FL_ROUNDS):
    metrics3 = server3.aggregate_round(clients3, r)
print(f"  SCAFFOLD final loss: {metrics3['avg_client_loss']:.6f}")

print("=== FL Training (FedAvg + GNN) ===")
Config.AGGREGATION_METHOD = "FedAvg"
global_model_gnn = create_model("GNN", input_dim, num_elements, config=Config)
server_gnn = FederatedServer(global_model_gnn, Config)
clients_gnn = [RISClient(i, create_model("GNN", input_dim, num_elements, config=Config), datasets[i], Config) for i in range(Config.NUM_TILES)]
for r in range(Config.FL_ROUNDS):
    metrics_gnn = server_gnn.aggregate_round(clients_gnn, r)
print(f"  GNN FedAvg final loss: {metrics_gnn['avg_client_loss']:.6f}")

# Test channel model directly
print("\n=== Channel Model Unit Tests ===")
ch = RicianChannel(num_elements=64, k_factor_db=10.0, num_paths=5)
result = ch.generate_channel(
    tx_pos=np.array([5, 10, 1.5]),
    rx_pos=np.array([[3, 3, 1], [7, 7, 1]]),
    ris_pos=np.array([5, 0, 1.5]),
    scenario="LoS"
)
print(f"  h_direct: {result['h_direct'].shape}, h_bs_ris: {result['h_bs_ris'].shape}, h_ris_user: {result['h_ris_user'].shape}")
assert result['h_direct'].shape == (2,)
assert result['h_bs_ris'].shape == (64,)
assert result['h_ris_user'].shape == (2, 64)

# Test CSI error
from src.channel_model import apply_csi_error, apply_phase_noise, quantize_phases
h_noisy = apply_csi_error(result['h_direct'], error_variance=0.01)
assert h_noisy.shape == result['h_direct'].shape
phases = np.random.uniform(0, 2*np.pi, 64)
noisy_p = apply_phase_noise(phases, noise_std_deg=5.0)
assert noisy_p.shape == phases.shape
quant_p = quantize_phases(phases, num_bits=2)
assert quant_p.shape == phases.shape
print("  CSI error, phase noise, quantization: OK")

print("\n=== ALL TESTS PASSED ===")

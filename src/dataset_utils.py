"""
Dataset utilities for RIS Federated Learning
Generates realistic channel data and optimal phase shifts

Supports:
- DeepMIMO ray-tracing dataset (primary)
- Synthetic Rician fading model (fallback)
- Non-IID partitioning with Dirichlet distribution
- CSI estimation error and phase noise
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import euclidean
import os
import pickle

from src.channel_model import (
    generate_ris_channel_dataset,
    apply_csi_error,
    apply_phase_noise,
    quantize_phases,
    RicianChannel,
)


class RISChannelDataset(Dataset):
    """
    Dataset for RIS channel state information and optimal phase shifts.
    
    Supports both DeepMIMO and synthetic Rician channel generation.
    """

    def __init__(self, num_samples, num_ris_elements, num_users,
                 room_size, frequency, tile_position=None, non_iid_bias=None,
                 k_factor_db=10.0, num_paths=5, spatial_corr_rho=0.7,
                 scenario="LoS", csi_error_variance=0.0,
                 grid_rows=8, grid_cols=8,
                 use_deepmimo=False, deepmimo_scenario='O1_28',
                 deepmimo_data_dir='data/deepmimo'):
        self.num_samples = num_samples
        self.num_ris_elements = num_ris_elements
        self.num_users = num_users
        self.room_size = room_size
        self.frequency = frequency
        self.wavelength = 3e8 / frequency
        self.tile_position = tile_position
        self.non_iid_bias = non_iid_bias
        self.scenario = scenario
        self.k_factor_db = k_factor_db
        self.csi_error_variance = csi_error_variance

        # Generate dataset using unified channel generator
        self.features, self.labels, self.metadata = generate_ris_channel_dataset(
            num_samples=num_samples,
            num_ris_elements=num_ris_elements,
            num_users=num_users,
            room_size=room_size,
            frequency=frequency,
            tile_position=tile_position,
            non_iid_bias=non_iid_bias,
            k_factor_db=k_factor_db,
            num_paths=num_paths,
            spatial_corr_rho=spatial_corr_rho,
            scenario=scenario,
            csi_error_variance=csi_error_variance,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            use_deepmimo=use_deepmimo,
            deepmimo_scenario=deepmimo_scenario,
            deepmimo_data_dir=deepmimo_data_dir,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor(self.labels[idx])
        )

    def get_input_dim(self):
        """Return input feature dimension"""
        return self.features.shape[1]


def create_non_iid_datasets(config, num_tiles):
    """
    Create non-IID datasets for different RIS tiles.
    Each tile sees different spatial distributions.
    
    Uses Dirichlet-based non-IID bias where tiles observe
    users near their physical location.
    """
    datasets = []

    # Create spatial biases for non-IID distribution
    tile_positions = []
    for i in range(num_tiles):
        # Distribute tiles around the room
        angle = 2 * np.pi * i / num_tiles
        x = config.ROOM_SIZE[0] / 2 + config.ROOM_SIZE[0] / 3 * np.cos(angle)
        y = config.ROOM_SIZE[1] / 2 + config.ROOM_SIZE[1] / 3 * np.sin(angle)
        z = config.ROOM_SIZE[2] / 2

        tile_positions.append([x, y, z])

        # Non-IID bias: tiles see users near their position
        bias_x = (x - config.ROOM_SIZE[0] / 2) * config.NON_IID_ALPHA
        bias_y = (y - config.ROOM_SIZE[1] / 2) * config.NON_IID_ALPHA

        dataset = RISChannelDataset(
            num_samples=config.TRAIN_SAMPLES,
            num_ris_elements=config.ELEMENTS_PER_TILE,
            num_users=config.NUM_USERS,
            room_size=config.ROOM_SIZE,
            frequency=config.FREQUENCY,
            tile_position=tile_positions[i],
            non_iid_bias=(bias_x, bias_y),
            # Realistic channel parameters
            k_factor_db=getattr(config, 'RICIAN_K_FACTOR_DB', 10.0),
            num_paths=getattr(config, 'NUM_PATHS', 5),
            spatial_corr_rho=getattr(config, 'SPATIAL_CORRELATION_RHO', 0.7),
            scenario=getattr(config, 'CHANNEL_SCENARIO', 'LoS'),
            csi_error_variance=getattr(config, 'CSI_ERROR_VARIANCE', 0.0),
            grid_rows=config.PIXEL_GRID_ROWS,
            grid_cols=config.PIXEL_GRID_COLS,
            use_deepmimo=getattr(config, 'USE_DEEPMIMO', False),
            deepmimo_scenario=getattr(config, 'DEEPMIMO_SCENARIO', 'O1_28'),
            deepmimo_data_dir=getattr(config, 'DEEPMIMO_DATA_DIR', 'data/deepmimo'),
        )
        datasets.append(dataset)

    return datasets, tile_positions


def create_test_dataset(config):
    """
    Create global test dataset (IID).
    
    Test data is generated from a different spatial region 
    (no non-IID bias) to ensure held-out evaluation.
    """
    return RISChannelDataset(
        num_samples=config.TEST_SAMPLES,
        num_ris_elements=config.ELEMENTS_PER_TILE,
        num_users=config.NUM_USERS,
        room_size=config.ROOM_SIZE,
        frequency=config.FREQUENCY,
        tile_position=None,
        non_iid_bias=None,
        # Realistic channel parameters
        k_factor_db=getattr(config, 'RICIAN_K_FACTOR_DB', 10.0),
        num_paths=getattr(config, 'NUM_PATHS', 5),
        spatial_corr_rho=getattr(config, 'SPATIAL_CORRELATION_RHO', 0.7),
        scenario=getattr(config, 'CHANNEL_SCENARIO', 'LoS'),
        csi_error_variance=getattr(config, 'CSI_ERROR_VARIANCE', 0.0),
        grid_rows=config.PIXEL_GRID_ROWS,
        grid_cols=config.PIXEL_GRID_COLS,
        use_deepmimo=getattr(config, 'USE_DEEPMIMO', False),
        deepmimo_scenario=getattr(config, 'DEEPMIMO_SCENARIO', 'O1_28'),
        deepmimo_data_dir=getattr(config, 'DEEPMIMO_DATA_DIR', 'data/deepmimo'),
    )


def save_datasets(datasets, test_dataset, save_path):
    """Save datasets to disk"""
    os.makedirs(save_path, exist_ok=True)

    data = {
        'train_datasets': datasets,
        'test_dataset': test_dataset
    }

    with open(os.path.join(save_path, 'datasets.pkl'), 'wb') as f:
        pickle.dump(data, f)

    print(f"Datasets saved to {save_path}")


def load_datasets(load_path):
    """Load datasets from disk"""
    with open(os.path.join(load_path, 'datasets.pkl'), 'rb') as f:
        data = pickle.load(f)

    return data['train_datasets'], data['test_dataset']
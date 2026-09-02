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
from utils.logger import logger
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


def expected_feature_dim(num_ris_elements, num_users):
    """
    Return the expected flat feature dimension for generated RIS datasets.

    The feature vector contains normalized user positions plus real and
    imaginary parts of the direct and cascaded channels:
        3U + 2 * (U + U * N)
    where U is user count and N is RIS elements per tile.
    """
    return int(num_users * (2 * num_ris_elements + 5))


def expected_label_dim(num_ris_elements):
    """Return the expected phase-label dimension for one RIS tile."""
    return int(num_ris_elements)


def validate_dataset_feature_dim(dataset, config, dataset_name="dataset"):
    """Validate that a dataset matches the active Config dimensions."""
    if dataset is None:
        raise ValueError(f"{dataset_name} is None")

    if not hasattr(dataset, "get_input_dim"):
        raise TypeError(f"{dataset_name} must provide get_input_dim()")

    expected = expected_feature_dim(config.ELEMENTS_PER_TILE, config.NUM_USERS)
    actual = dataset.get_input_dim()
    if actual != expected:
        raise ValueError(
            f"{dataset_name} feature dimension mismatch: expected {expected} "
            f"for NUM_USERS={config.NUM_USERS}, ELEMENTS_PER_TILE={config.ELEMENTS_PER_TILE}; "
            f"got {actual}. Regenerate datasets after changing geometry/user settings."
        )

    dataset_users = getattr(dataset, "num_users", None)
    if dataset_users is not None and dataset_users != config.NUM_USERS:
        raise ValueError(
            f"{dataset_name} user-count mismatch: expected NUM_USERS={config.NUM_USERS}; "
            f"dataset was built with num_users={dataset_users}."
        )

    dataset_elements = getattr(dataset, "num_ris_elements", None)
    if dataset_elements is not None and dataset_elements != config.ELEMENTS_PER_TILE:
        raise ValueError(
            f"{dataset_name} RIS-element mismatch: expected ELEMENTS_PER_TILE="
            f"{config.ELEMENTS_PER_TILE}; dataset was built with "
            f"num_ris_elements={dataset_elements}."
        )

    labels = getattr(dataset, "labels", None)
    if labels is not None:
        expected_labels = expected_label_dim(config.ELEMENTS_PER_TILE)
        if labels.ndim != 2 or labels.shape[1] != expected_labels:
            raise ValueError(
                f"{dataset_name} label dimension mismatch: expected labels with "
                f"shape (*, {expected_labels}); got {labels.shape}."
            )

    return True


def validate_dataset_collection(train_datasets, test_dataset, config, expected_num_tiles=None):
    """
    Validate train/test datasets before model creation.

    Returns the single input dimension that should be used to instantiate models.
    """
    if not train_datasets:
        raise ValueError("train_datasets must contain at least one dataset")

    if expected_num_tiles is not None and len(train_datasets) != expected_num_tiles:
        raise ValueError(
            f"train_datasets tile-count mismatch: expected {expected_num_tiles}; "
            f"got {len(train_datasets)}."
        )

    input_dim = None
    for idx, dataset in enumerate(train_datasets):
        name = f"train_datasets[{idx}]"
        validate_dataset_feature_dim(dataset, config, name)
        dim = dataset.get_input_dim()
        if input_dim is None:
            input_dim = dim
        elif dim != input_dim:
            raise ValueError(
                f"{name} feature dimension mismatch: expected {input_dim}; got {dim}."
            )

    if test_dataset is not None:
        validate_dataset_feature_dim(test_dataset, config, "test_dataset")
        test_dim = test_dataset.get_input_dim()
        if test_dim != input_dim:
            raise ValueError(
                f"test_dataset feature dimension mismatch: expected {input_dim}; "
                f"got {test_dim}."
            )

    return input_dim


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
        expected = expected_feature_dim(num_ris_elements, num_users)
        if self.features.shape[1] != expected:
            raise ValueError(
                f"Generated feature dimension mismatch: expected {expected}, "
                f"got {self.features.shape[1]}"
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

    logger.info(f"Datasets saved to {save_path}")


def load_datasets(load_path):
    """Load datasets from disk"""
    with open(os.path.join(load_path, 'datasets.pkl'), 'rb') as f:
        data = pickle.load(f)

    return data['train_datasets'], data['test_dataset']

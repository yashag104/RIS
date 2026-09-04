"""
Dataset utilities for RIS Federated Learning
Generates realistic channel data and optimal phase shifts

Supports:
- DeepMIMO ray-tracing dataset (primary)
- Synthetic Rician fading model (fallback)
- Non-IID partitioning with Dirichlet distribution
- CSI estimation error and phase noise
"""

import os
import pickle
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.channel_model import (
    generate_ris_channel_dataset,
)
from utils.logger import logger


def expected_feature_dim(num_ris_elements: int, num_users: int) -> int:
    """
    Return the expected flat feature dimension for generated RIS datasets.

    The feature vector contains normalized user positions plus real and
    imaginary parts of the direct and cascaded channels:
        3U + 2 * (U + U * N)
    where U is user count and N is RIS elements per tile.
    """
    return int(num_users * (2 * num_ris_elements + 5))


def expected_label_dim(num_ris_elements: int) -> int:
    """Return the expected phase-label dimension for one RIS tile."""
    return int(num_ris_elements)


def validate_dataset_feature_dim(dataset, config, dataset_name: str = "dataset") -> bool:
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


def validate_dataset_collection(train_datasets: list, test_dataset, config, expected_num_tiles: int | None = None) -> int:
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

    def __init__(self, num_samples: int, num_ris_elements: int, num_users: int,
                 room_size, frequency: float, tile_position=None, non_iid_bias=None,
                 k_factor_db: float = 10.0, num_paths: int = 5,
                 spatial_corr_rho: float = 0.7, scenario: str = "LoS",
                 csi_error_variance: float = 0.0, grid_rows: int = 8,
                 grid_cols: int = 8, use_deepmimo: bool = False,
                 deepmimo_scenario: str = 'O1_28',
                 deepmimo_data_dir: str = 'data/deepmimo',
                 direct_link_blockage_db: float = 30.0):
        """Generate a synthetic RIS dataset using the configured channel model.

        Args:
            num_samples: Number of (feature, label) training examples to generate.
            num_ris_elements: Total number of phase-controllable RIS elements.
            num_users: Number of single-antenna users in the scene.
            room_size: 3-tuple ``(x, y, z)`` of room dimensions in metres.
            frequency: Carrier frequency in Hz (e.g. ``28e9`` for 28 GHz).
            tile_position: Optional (x, y) position of this tile in the room.
                When set, channel statistics reflect tile-specific geometry.
            non_iid_bias: Optional bias vector introducing non-IID heterogeneity
                across tiles (simulates spatial data skew).
            k_factor_db: Rician K-factor in dB; controls LoS dominance.
            num_paths: Number of multi-path components in the channel model.
            spatial_corr_rho: Spatial correlation coefficient ρ ∈ [0, 1].
            scenario: Channel scenario label; e.g. ``'LoS'`` or ``'NLoS'``.
            csi_error_variance: Variance of additive Gaussian CSI estimation noise.
            grid_rows: Number of pixel rows per RIS tile (must satisfy
                ``grid_rows × grid_cols == num_ris_elements``).
            grid_cols: Number of pixel columns per RIS tile.
            use_deepmimo: If ``True``, use DeepMIMO ray-tracing data instead of
                the synthetic Rician model (requires DeepMIMOv3 package).
            deepmimo_scenario: DeepMIMO scenario name (e.g. ``'O1_28'``).
            direct_link_blockage_db: Excess attenuation on the obstructed
                BS->user direct path, in dB. See RicianChannel.
            deepmimo_data_dir: Path to the directory containing DeepMIMO
                scenario data files.

        Raises:
            ValueError: If the generated feature dimension does not match the
                expected value computed from ``num_ris_elements`` and ``num_users``.
        """
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
            direct_link_blockage_db=direct_link_blockage_db,
        )
        expected = expected_feature_dim(num_ris_elements, num_users)
        if self.features.shape[1] != expected:
            raise ValueError(
                f"Generated feature dimension mismatch: expected {expected}, "
                f"got {self.features.shape[1]}"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int):
        """Return the ``(features, labels)`` pair at the given index.

        Args:
            idx: Sample index in ``[0, len(self))``.

        Returns:
            Tuple of ``(torch.FloatTensor, torch.FloatTensor)`` for the
            feature vector and phase-shift label vector respectively.
        """
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor(self.labels[idx])
        )

    def get_input_dim(self) -> int:
        """Return input feature dimension"""
        return self.features.shape[1]


def create_non_iid_datasets(config, num_tiles: int) -> tuple[list, list]:
    """Create spatially non-IID datasets for a fleet of RIS tiles.

    Each tile's dataset is generated from a distinct spatial location in the
    room, with a non-IID bias proportional to the tile's displacement from
    the room centre, simulating real heterogeneity in user distributions.

    Args:
        config: :class:`~config.Config` instance with room and FL settings.
        num_tiles: Number of RIS tiles (one dataset is generated per tile).

    Returns:
        Tuple of:
        - ``datasets``: List of :class:`RISChannelDataset` instances, one per tile.
        - ``tile_positions``: List of ``[x, y, z]`` position vectors for each tile.
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
            direct_link_blockage_db=getattr(config, 'DIRECT_LINK_BLOCKAGE_DB', 30.0),
        )
        datasets.append(dataset)

    return datasets, tile_positions


def create_test_dataset(config) -> "RISChannelDataset":
    """Create a global IID test dataset from a randomly distributed user population.

    Test data is generated without any non-IID spatial bias so that evaluation
    reflects the full environment distribution.

    Args:
        config: :class:`~config.Config` instance with dataset and channel settings.

    Returns:
        A :class:`RISChannelDataset` with ``config.TEST_SAMPLES`` samples.
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
        direct_link_blockage_db=getattr(config, 'DIRECT_LINK_BLOCKAGE_DB', 30.0),
    )


def save_datasets(datasets: list, test_dataset, save_path: str) -> None:
    """Save datasets to disk.

    Args:
        datasets: List of training :class:`RISChannelDataset` instances.
        test_dataset: Global test :class:`RISChannelDataset` instance.
        save_path: Directory path where ``datasets.pkl`` will be written.
    Returns:
        None
    """
    os.makedirs(save_path, exist_ok=True)

    data = {
        'train_datasets': datasets,
        'test_dataset': test_dataset
    }

    with open(os.path.join(save_path, 'datasets.pkl'), 'wb') as f:
        pickle.dump(data, f)

    logger.info(f"Datasets saved to {save_path}")


def load_datasets(load_path: str) -> tuple[list, Any]:
    """Load datasets from disk.

    Args:
        load_path: Directory containing a previously saved ``datasets.pkl``.
    Returns:
        Tuple of ``(train_datasets, test_dataset)``.
    """
    with open(os.path.join(load_path, 'datasets.pkl'), 'rb') as f:
        data = pickle.load(f)

    return data['train_datasets'], data['test_dataset']

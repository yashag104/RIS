"""Channels for ONE contiguous tiled RIS, generated for the whole aperture at once.

The legacy generator (:func:`src.channel_model.generate_multi_tile_channels`)
placed the 16 tiles on a 3.3 m circle around the room and gave each its own
independent NLoS draw, so the "surface" was really 16 separate panels metres
apart. That contradicts the system model -- one 4x4-tiled metasurface whose
tiles are wired together by an on-die interconnect -- and it is what made the
array-gain curve saturate: the far-side panels were metres further from the BS.

Here the surface is a single ``(TR*PR) x (TC*PC)`` uniform planar array at
lambda/2 pitch (32 x 32 = 1024 elements, ~17 cm across at 28 GHz), mounted on
the ``y = 0`` wall. Tiles are contiguous ``PR x PC`` blocks of it. Every channel
is generated for the full aperture and then sliced, so the tiles see the same
scatterers and their contributions sum coherently by construction.

Physics, per scene:

* **LoS**, exact spherical wavefront per element. Depending on position, users
  in a 10 m room can be in the radiative near field of this aperture; exact
  distances avoid assuming a far-field plane wave.
* **NLoS**, ``num_paths`` planar waves per hop with random directions in the
  half-space in front of the wall and exponentially decaying CN gains. Spatial
  correlation across the aperture comes from this path geometry. (The legacy
  model instead multiplied the *whole* channel -- LoS included -- by a Cholesky
  factor, which scrambles the deterministic LoS wavefront.)
* Close-in path loss (exponent 2.5 per RIS hop, 3.5 for the direct link), the
  same per-element aperture gain ``4 pi A / lambda^2`` per hop, Rician mixing per
  hop, and an explicit direct-link blockage -- all identical to the legacy model
  so link budgets stay comparable.

Arrays are returned tile-major: ``cascade[s, t, n]`` is element ``n`` (row-major
within the tile) of tile ``t`` (row-major over the tile grid).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_LIGHT = 3e8


@dataclass(frozen=True)
class SurfaceGeometry:
    """Where every element of the tiled surface sits."""

    tile_rows: int = 4
    tile_cols: int = 4
    pixel_rows: int = 8
    pixel_cols: int = 8
    frequency: float = 28e9
    spacing_wavelengths: float = 0.5
    center: tuple = (5.0, 0.0, 1.5)

    def __post_init__(self):
        if min(self.tile_rows, self.tile_cols, self.pixel_rows, self.pixel_cols) < 1:
            raise ValueError("Surface grid dimensions must be positive")
        if self.frequency <= 0 or self.spacing_wavelengths <= 0:
            raise ValueError("Frequency and element spacing must be positive")

    @property
    def wavelength(self) -> float:
        return C_LIGHT / self.frequency

    @property
    def spacing(self) -> float:
        return self.spacing_wavelengths * self.wavelength

    @property
    def num_tiles(self) -> int:
        return self.tile_rows * self.tile_cols

    @property
    def elements_per_tile(self) -> int:
        return self.pixel_rows * self.pixel_cols

    @property
    def total_elements(self) -> int:
        return self.num_tiles * self.elements_per_tile

    @property
    def aperture_m(self) -> tuple[float, float]:
        """(width, height) of the full surface in metres."""
        return (self.tile_cols * self.pixel_cols * self.spacing,
                self.tile_rows * self.pixel_rows * self.spacing)

    def global_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """(row, col) of every element on the full grid, shape (T, Ne) each."""
        t = np.arange(self.num_tiles)
        n = np.arange(self.elements_per_tile)
        tr, tc = t // self.tile_cols, t % self.tile_cols
        pr, pc = n // self.pixel_cols, n % self.pixel_cols
        rows = tr[:, None] * self.pixel_rows + pr[None, :]
        cols = tc[:, None] * self.pixel_cols + pc[None, :]
        return rows, cols

    def element_positions(self) -> np.ndarray:
        """Element coordinates, shape (T, Ne, 3). The surface lies in the x-z plane."""
        rows, cols = self.global_indices()
        n_rows = self.tile_rows * self.pixel_rows
        n_cols = self.tile_cols * self.pixel_cols
        cx, cy, cz = self.center
        x = cx + (cols - (n_cols - 1) / 2) * self.spacing
        z = cz + (rows - (n_rows - 1) / 2) * self.spacing
        y = np.full_like(x, cy, dtype=float)
        return np.stack([x, y, z], axis=-1)

    def tile_centers(self) -> np.ndarray:
        """Tile centre coordinates, shape (T, 3)."""
        return self.element_positions().mean(axis=1)

    def tile_grid_coords(self) -> np.ndarray:
        """Normalised (row, col) of each tile in [-1, 1], shape (T, 2)."""
        t = np.arange(self.num_tiles)
        r = t // self.tile_cols
        c = t % self.tile_cols
        rn = 2 * r / max(self.tile_rows - 1, 1) - 1
        cn = 2 * c / max(self.tile_cols - 1, 1) - 1
        return np.stack([rn, cn], axis=-1).astype(np.float32)


@dataclass(frozen=True)
class SceneConfig:
    """Everything about the propagation scene that is not the surface itself."""

    bs_position: tuple = (5.0, 10.0, 1.5)
    user_low: tuple = (0.5, 1.0, 1.0)
    user_high: tuple = (9.5, 9.5, 2.0)
    k_factor_db: float = 10.0
    num_paths: int = 5
    ris_path_loss_exponent: float = 2.5
    direct_path_loss_exponent: float = 3.5
    direct_link_blockage_db: float = 30.0
    element_gain_enabled: bool = True


def close_in_path_loss(distance, wavelength: float, exponent: float) -> np.ndarray:
    """Close-in free-space-reference path loss, linear power, d_ref = 1 m.

    Same formula as ``RicianChannel._compute_path_loss`` so link budgets match.
    """
    d = np.maximum(np.asarray(distance, dtype=float), 0.1)
    return (wavelength / (4 * np.pi)) ** 2 * d ** (-exponent)


def _random_front_directions(rng, shape) -> np.ndarray:
    """Unit vectors uniformly distributed over the half-space y > 0."""
    v = rng.standard_normal(shape + (3,))
    v[..., 1] = np.abs(v[..., 1])
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _nlos_field(rng, positions_flat, wavelength, num_scenes, num_paths) -> np.ndarray:
    """Sum of ``num_paths`` planar waves across the aperture, unit mean power.

    Returns shape (S, N_total). Path ``p`` has power ``exp(-p)`` before the
    sum is normalised, matching the legacy exponential decay.
    """
    k = 2 * np.pi / wavelength
    dirs = _random_front_directions(rng, (num_scenes, num_paths))          # (S, P, 3)
    p_idx = np.arange(num_paths)
    amp = np.exp(-0.5 * p_idx)
    amp = amp / np.sqrt(np.sum(amp ** 2))
    gains = (rng.standard_normal((num_scenes, num_paths))
             + 1j * rng.standard_normal((num_scenes, num_paths))) / np.sqrt(2)
    gains = gains * amp[None, :]
    rel = positions_flat - positions_flat.mean(axis=0, keepdims=True)       # (N, 3)
    phase = k * np.einsum("spd,nd->spn", dirs, rel)                         # (S, P, N)
    return np.einsum("sp,spn->sn", gains, np.exp(1j * phase))


def generate_surface_channels(
    num_scenes: int,
    geometry: SurfaceGeometry,
    scene: SceneConfig,
    rng: np.random.Generator,
    batch: int = 256,
) -> dict:
    """Draw ``num_scenes`` scenes for the whole surface.

    Returns a dict of arrays:

    * ``h_direct``  (S,)        complex, BS -> user (blocked)
    * ``cascade``   (S, T, Ne)  complex, BS -> element -> user, per element
    * ``cascade_los`` (S, T, Ne) complex, the LoS-only part of the cascade,
      i.e. what a location-aware controller could compute from geometry
    * ``h_direct_los`` (S,)     complex, LoS part of the direct link
    * ``user_pos``  (S, 3)
    """
    if num_scenes < 1 or batch < 1 or scene.num_paths < 1:
        raise ValueError("Scene count, batch size and path count must be positive")
    lam = geometry.wavelength
    k = 2 * np.pi / lam
    pos = geometry.element_positions()                     # (T, Ne, 3)
    T, Ne = pos.shape[:2]
    pos_flat = pos.reshape(-1, 3)
    center = pos_flat.mean(axis=0)
    bs = np.asarray(scene.bs_position, dtype=float)

    K = 10 ** (scene.k_factor_db / 10)
    w_los, w_nlos = np.sqrt(K / (K + 1)), np.sqrt(1 / (K + 1))
    g_amp = 1.0
    if scene.element_gain_enabled:
        g_amp = np.sqrt(4 * np.pi * geometry.spacing ** 2 / lam ** 2)
    blockage = 10 ** (-scene.direct_link_blockage_db / 10)
    ple = scene.ris_path_loss_exponent

    # BS -> element LoS is fixed: the BS and the surface do not move.
    d_bs = np.linalg.norm(pos_flat - bs, axis=1)                            # (N,)
    h_bs_los = np.sqrt(close_in_path_loss(d_bs, lam, ple)) * np.exp(-1j * k * d_bs)
    pl_bs_center = close_in_path_loss(np.linalg.norm(center - bs), lam, ple)

    out = {
        "h_direct": np.empty(num_scenes, dtype=np.complex128),
        "h_direct_los": np.empty(num_scenes, dtype=np.complex128),
        "cascade": np.empty((num_scenes, T, Ne), dtype=np.complex64),
        "cascade_los": np.empty((num_scenes, T, Ne), dtype=np.complex64),
        "user_pos": np.empty((num_scenes, 3), dtype=np.float64),
    }

    lo, hi = np.asarray(scene.user_low), np.asarray(scene.user_high)
    for start in range(0, num_scenes, batch):
        s = min(batch, num_scenes - start)
        sl = slice(start, start + s)
        user = rng.uniform(lo, hi, size=(s, 3))

        # Direct link, Rician, blocked.
        d_dir = np.linalg.norm(user - bs, axis=1)
        pl_dir = close_in_path_loss(d_dir, lam, scene.direct_path_loss_exponent) * blockage
        hd_los = np.sqrt(pl_dir) * np.exp(-1j * k * d_dir)
        hd_nlos = np.sqrt(pl_dir) * (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)
        out["h_direct"][sl] = w_los * hd_los + w_nlos * hd_nlos
        out["h_direct_los"][sl] = w_los * hd_los

        # Element -> user LoS, exact per-element distance.
        d_u = np.linalg.norm(pos_flat[None, :, :] - user[:, None, :], axis=2)  # (s, N)
        h_u_los = np.sqrt(close_in_path_loss(d_u, lam, ple)) * np.exp(-1j * k * d_u)
        pl_u_center = close_in_path_loss(np.linalg.norm(user - center, axis=1), lam, ple)

        h_bs_nlos = np.sqrt(pl_bs_center) * _nlos_field(rng, pos_flat, lam, s, scene.num_paths)
        h_u_nlos = np.sqrt(pl_u_center)[:, None] * _nlos_field(rng, pos_flat, lam, s, scene.num_paths)

        h_bs = w_los * h_bs_los[None, :] + w_nlos * h_bs_nlos
        h_u = w_los * h_u_los + w_nlos * h_u_nlos
        casc = (g_amp ** 2) * h_bs * h_u
        casc_los = (g_amp ** 2) * (w_los * h_bs_los[None, :]) * (w_los * h_u_los)

        out["cascade"][sl] = casc.reshape(s, T, Ne)
        out["cascade_los"][sl] = casc_los.reshape(s, T, Ne)
        out["user_pos"][sl] = user

    return out


def geometry_from_config(config) -> SurfaceGeometry:
    return SurfaceGeometry(
        tile_rows=config.TILE_GRID_ROWS,
        tile_cols=config.TILE_GRID_COLS,
        pixel_rows=config.PIXEL_GRID_ROWS,
        pixel_cols=config.PIXEL_GRID_COLS,
        frequency=config.FREQUENCY,
        center=tuple(getattr(config, "SURFACE_CENTER", (5.0, 0.0, 1.5))),
    )


def scene_from_config(config) -> SceneConfig:
    return SceneConfig(
        bs_position=tuple(getattr(config, "BS_POSITION", (5.0, 10.0, 1.5))),
        user_low=tuple(getattr(config, "USER_REGION_LOW", (0.5, 1.0, 1.0))),
        user_high=tuple(getattr(config, "USER_REGION_HIGH", (9.5, 9.5, 2.0))),
        k_factor_db=config.RICIAN_K_FACTOR_DB,
        num_paths=config.NUM_PATHS,
        ris_path_loss_exponent=config.PATH_LOSS_EXPONENT,
        direct_link_blockage_db=config.DIRECT_LINK_BLOCKAGE_DB,
        element_gain_enabled=config.RIS_ELEMENT_GAIN_ENABLED,
    )


def surface_datasets(channels: dict, geometry: SurfaceGeometry,
                     error_variance: float = 0.0, seed: int = 0) -> list:
    """Adapt whole-aperture scenes to the existing tile learner.

    This is an explicitly *supplied-CSI* diagnostic, not a passive sensing
    model. Noise is added once to the cascaded channel, with epsilon equal to
    its mean-square error divided by mean channel power. The direct estimate
    is shared across all tiles. Reusing a seed couples noise across epsilon.
    User locations are not supplied as privileged side information.
    """
    from src.dataset_utils import RISChannelDataset

    if error_variance < 0:
        raise ValueError("CSI error variance must be nonnegative")
    rng = np.random.default_rng(seed)
    c = channels["cascade"]
    hd = channels["h_direct"]
    zc = (rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape)) / np.sqrt(2)
    zd = (rng.standard_normal(hd.shape) + 1j * rng.standard_normal(hd.shape)) / np.sqrt(2)
    ce = c + np.sqrt(error_variance * np.mean(np.abs(c) ** 2)) * zc
    he = hd + np.sqrt(error_variance * np.mean(np.abs(hd) ** 2)) * zd
    direct_scale = np.maximum(np.abs(he) / np.sqrt(2), 1e-30)
    result = []
    for t in range(geometry.num_tiles):
        records = [{"h_direct": hd[s:s+1], "h_ris_user": c[s, t:t+1],
                    "h_bs_ris": np.ones(geometry.elements_per_tile, dtype=complex),
                    "user_positions": np.zeros((1, 3))} for s in range(len(hd))]
        ds = RISChannelDataset.from_channels(records, geometry.elements_per_tile, 1)
        ct = ce[:, t]
        scale = np.maximum(np.sqrt(np.mean(np.abs(ct) ** 2, axis=1) / 2), 1e-30)
        ds.features = np.concatenate([
            np.zeros((len(hd), 3)), (he.real / direct_scale)[:, None],
            ct.real / scale[:, None], (he.imag / direct_scale)[:, None],
            ct.imag / scale[:, None]], axis=1).astype(np.float32)
        ds.phase_offset = np.angle(he)
        # Labels are training truth; estimates enter features only.
        ds.labels = np.mod(-np.angle(c[:, t]), 2 * np.pi).astype(np.float32)
        for s, m in enumerate(ds.metadata):
            m["H_direct_est"] = he[s:s+1]
            m["H_ris_est"] = ce[s, t:t+1]
            m["h_bs_ris_est"] = np.ones(geometry.elements_per_tile, dtype=complex)
            m["phase_offset"] = float(ds.phase_offset[s])
        result.append(ds)
    return result

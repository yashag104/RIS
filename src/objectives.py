"""Differentiable communication objectives for RIS phase prediction.

The original training signal was an MSE distance to a precomputed MRC phase
target for a single hardcoded user (``target_user = 0``). That has two problems:

* MSE on phases is not the quantity anyone cares about. Two phase vectors with
  the same MSE can achieve very different SNR, because the error that matters is
  the *coherence* of the summed reflected paths, not per-element angular error.
* A single-user target cannot express a multi-user trade-off at all, so the
  network could never learn to serve more than one user.

This module scores what the system actually delivers -- received SNR, or the
weighted sum-rate across every user -- directly from the predicted phases and
the true channels, in a form autograd can differentiate.

Shapes throughout:
    phases    (B, N)        predicted RIS phase shifts, radians
    h_direct  (B, U, 2)     direct BS->user channel, (real, imag) trailing axis
    h_cascade (B, U, N, 2)  cascaded BS->RIS->user channel, same convention
"""

from __future__ import annotations

import torch

# Fraction of another user's received power that leaks into this user's SINR.
# Zero by default: a single-antenna BS with one shared RIS serves users on
# orthogonal resources, so there is no co-channel interference term. Must match
# Config.CROSS_TALK_FACTOR and the evaluation in
# experiments.baselines_multiuser -- an objective that assumes a different
# interference model than the evaluation is optimizing something nobody measures.
DEFAULT_CROSS_TALK_FACTOR = 0.0


def received_power(
    phases: torch.Tensor,
    h_direct: torch.Tensor,
    h_cascade: torch.Tensor,
) -> torch.Tensor:
    """Received signal power per user, up to the transmit-power scaling.

    Computes ``|h_direct_u + sum_n h_cascade_un * exp(j*theta_n)|^2`` without
    complex tensors, so the graph stays differentiable on every backend.

    Returns:
        Tensor of shape ``(B, U)``.
    """
    cos_t = torch.cos(phases).unsqueeze(1)  # (B, 1, N)
    sin_t = torch.sin(phases).unsqueeze(1)

    casc_re = h_cascade[..., 0]  # (B, U, N)
    casc_im = h_cascade[..., 1]

    # Complex multiply h_cascade * exp(j*theta), then sum over elements.
    refl_re = (casc_re * cos_t - casc_im * sin_t).sum(dim=-1)  # (B, U)
    refl_im = (casc_re * sin_t + casc_im * cos_t).sum(dim=-1)

    total_re = h_direct[..., 0] + refl_re
    total_im = h_direct[..., 1] + refl_im

    return total_re ** 2 + total_im ** 2


def sum_rate(
    phases: torch.Tensor,
    h_direct: torch.Tensor,
    h_cascade: torch.Tensor,
    noise_power: float,
    tx_power: float,
    user_weights: torch.Tensor | None = None,
    cross_talk: float = DEFAULT_CROSS_TALK_FACTOR,
) -> torch.Tensor:
    """Weighted sum-rate (bits/s/Hz) per batch element, shape ``(B,)``.

    Interference uses the same cross-talk model as the multi-user evaluation:
    every other user's received power leaks in scaled by ``cross_talk``. Set
    ``cross_talk=0`` for the interference-free single-cell case.
    """
    powers = tx_power * received_power(phases, h_direct, h_cascade)  # (B, U)

    if cross_talk:
        interference = cross_talk * (powers.sum(dim=1, keepdim=True) - powers)
    else:
        interference = torch.zeros_like(powers)

    sinr = powers / (noise_power + interference)
    rates = torch.log2(1.0 + sinr)

    if user_weights is not None:
        rates = rates * user_weights.to(rates.device).view(1, -1)

    return rates.sum(dim=1)


class SumRateLoss(torch.nn.Module):
    """Negative weighted sum-rate, for gradient *descent*.

    The raw sum-rate is scale-free in a bad way for optimization: at these path
    losses SINR is far below 1, so ``log2(1+x) ~ x/ln2`` and the loss lives near
    zero with tiny gradients. Normalizing by the genie-optimal rate for the same
    batch keeps the loss O(1) and makes it directly interpretable -- 0.0 means
    the prediction matched the genie bound, 1.0 means it achieved nothing.
    """

    def __init__(
        self,
        noise_power: float,
        tx_power: float,
        user_weights: torch.Tensor | None = None,
        cross_talk: float = DEFAULT_CROSS_TALK_FACTOR,
        normalize: bool = True,
    ):
        super().__init__()
        self.noise_power = noise_power
        self.tx_power = tx_power
        self.cross_talk = cross_talk
        self.normalize = normalize
        self.register_buffer(
            "user_weights",
            user_weights if user_weights is not None else torch.empty(0),
        )

    def _weights(self) -> torch.Tensor | None:
        return self.user_weights if self.user_weights.numel() else None

    def forward(
        self,
        phases: torch.Tensor,
        h_direct: torch.Tensor,
        h_cascade: torch.Tensor,
    ) -> torch.Tensor:
        achieved = sum_rate(
            phases, h_direct, h_cascade,
            self.noise_power, self.tx_power, self._weights(), self.cross_talk,
        )

        if not self.normalize:
            return -achieved.mean()

        # Genie bound: align every reflected path with the direct path, per user.
        # Only exactly achievable for one user at a time, so for U > 1 this is a
        # strict upper bound rather than a reachable target -- which is fine, it
        # is used purely as a per-batch normalizer.
        with torch.no_grad():
            bound = _genie_sum_rate(
                h_direct, h_cascade, self.noise_power, self.tx_power,
                self._weights(), self.cross_talk,
            )
            bound = bound.clamp_min(1e-12)

        return (1.0 - achieved / bound).mean()


def _genie_sum_rate(
    h_direct: torch.Tensor,
    h_cascade: torch.Tensor,
    noise_power: float,
    tx_power: float,
    user_weights: torch.Tensor | None,
    cross_talk: float,
) -> torch.Tensor:
    """Upper bound on sum-rate: coherent combining, evaluated per user.

    Each user's own MRC solution is applied in turn, so no single phase vector
    realizes this jointly when U > 1.
    """
    direct_mag = torch.sqrt(h_direct[..., 0] ** 2 + h_direct[..., 1] ** 2)  # (B, U)
    casc_mag = torch.sqrt(h_cascade[..., 0] ** 2 + h_cascade[..., 1] ** 2)  # (B, U, N)

    # Perfect alignment: amplitudes add.
    powers = tx_power * (direct_mag + casc_mag.sum(dim=-1)) ** 2

    if cross_talk:
        interference = cross_talk * (powers.sum(dim=1, keepdim=True) - powers)
    else:
        interference = torch.zeros_like(powers)

    rates = torch.log2(1.0 + powers / (noise_power + interference))
    if user_weights is not None:
        rates = rates * user_weights.to(rates.device).view(1, -1)
    return rates.sum(dim=1)


def build_objective(config):
    """Return ``(loss_module, needs_channels)`` for the configured objective.

    ``needs_channels`` tells the caller whether the dataset must yield channel
    tensors alongside features and labels.
    """
    objective = getattr(config, 'TRAINING_OBJECTIVE', 'mse').lower()

    if objective == 'mse':
        return torch.nn.MSELoss(), False

    if objective in ('snr', 'sumrate', 'sum_rate'):
        noise_power = 10 ** ((config.NOISE_POWER_DBM - 30) / 10)
        tx_power = 10 ** ((config.TX_POWER_DBM - 30) / 10)

        weights = getattr(config, 'SUM_RATE_USER_WEIGHTS', None)
        weight_tensor = torch.tensor(weights, dtype=torch.float32) if weights else None

        # 'snr' is the single-user special case: no interference term.
        cross_talk = 0.0 if objective == 'snr' else getattr(
            config, 'CROSS_TALK_FACTOR', DEFAULT_CROSS_TALK_FACTOR
        )

        return SumRateLoss(
            noise_power=noise_power,
            tx_power=tx_power,
            user_weights=weight_tensor,
            cross_talk=cross_talk,
        ), True

    raise ValueError(
        f"Unknown TRAINING_OBJECTIVE {objective!r}; expected 'mse', 'snr' or 'sumrate'"
    )

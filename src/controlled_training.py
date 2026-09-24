"""Controlled SISO learner comparisons; held-out validation never uses test data.

All learners start from the same weights and use Adam with a constant learning
rate, fixed batch size, and the same local genie-normalized rate objective.
FedAvg resets optimizer state on each broadcast (explicit experimental choice).
Both centralized budgets are reported: one client's serial steps and the sum
of all client steps. Those are different notions of matched computation.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch


def tensor_data(datasets, device):
    """Materialize local features and training truth; no test channels here."""
    return [tuple(torch.as_tensor(a, device=device) for a in (
        d.features.astype(np.float32), d.h_direct[:, 0].astype(np.complex64),
        d.h_cascade[:, 0].astype(np.complex64),
        d.phase_offset.astype(np.float32))) for d in datasets]


def rate_loss(model, data, indices, rho, absolute_phases=False):
    x, hd, c, offset = (a[indices] for a in data)
    comp = model.forward_components(x)
    phase = torch.atan2(comp[..., 1], comp[..., 0])
    if not absolute_phases:
        phase = phase + offset[:, None]
    power = (hd + (c * torch.exp(1j * phase)).sum(-1)).abs().square()
    bound = (hd.abs() + c.abs().sum(-1)).square()
    return (1 - torch.log1p(rho * power) / torch.log1p(rho * bound).clamp_min(1e-20)).mean()


def validation_loss(model, data, rho, absolute_phases=False):
    model.eval()
    with torch.no_grad():
        # Avoid materializing a large GAT activation for the entire validation
        # split. Accumulate by sample count so a short final batch is not biased.
        values = []
        for d in data:
            n = len(d[0])
            total = sum(float(rate_loss(model, d, slice(i, i+64), rho, absolute_phases)) *
                        min(64, n-i) for i in range(0, n, 64))
            values.append(total / n)
    return float(np.mean(values))


def train_steps(model, data, steps, batch_size, lr, rng, rho,
                absolute_phases=False, optimizer=None):
    """Exactly ``steps`` updates; uniform minibatches, including small datasets."""
    opt = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(steps):
        idx = torch.as_tensor(rng.choice(len(data[0]), min(batch_size, len(data[0])),
                                        replace=False), device=data[0].device)
        opt.zero_grad(set_to_none=True)
        loss = rate_loss(model, data, idx, rho, absolute_phases)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)), opt


def plateau(history, patience, min_delta):
    """No validation improvement of min_delta during the last patience rounds."""
    if len(history) <= patience:
        return False
    return min(history[-patience:]) >= min(history[:-patience]) - min_delta


def train_comparison(fresh_model, train, validation, config, *,
                     absolute_phases=False, progress=None):
    """FedAvg, fixed-round checkpoints, local-only, and two pooled budgets.

    Returns models plus explicit step, traffic, convergence, and selection
    records. A validation plateau is a stopping rule, not proof of optimality.
    """
    device = config.DEVICE
    td, vd = tensor_data(train, device), tensor_data(validation, device)
    initial = fresh_model().to(device)
    global_model = copy.deepcopy(initial)
    worker = copy.deepcopy(initial)
    lr = config.GNN_LEARNING_RATE if config.MODEL_TYPE == "GNN" else config.LEARNING_RATE
    rho = 10 ** ((config.TX_POWER_DBM - config.NOISE_POWER_DBM) / 10)
    per_round = config.LOCAL_EPOCHS * math.ceil(len(train[0]) / config.BATCH_SIZE)
    rng = np.random.default_rng(config.SEED + 100)
    models, histories, train_losses = {}, [], []
    best_loss, best_round, best_weights = math.inf, 0, None
    start = time.perf_counter()
    for r in range(1, config.FL_ROUNDS + 1):
        weights = global_model.state_dict()
        accum = {k: torch.zeros_like(v) for k, v in weights.items()}
        losses = []
        for d in td:
            worker.load_state_dict(weights)
            loss, _ = train_steps(worker, d, per_round, config.BATCH_SIZE, lr, rng, rho,
                                  absolute_phases)
            losses.append(loss)
            for k, v in worker.state_dict().items():
                if v.is_floating_point():
                    accum[k] += v / len(td)
                else:
                    accum[k] = v.clone()
        global_model.load_state_dict(accum)
        val = validation_loss(global_model, vd, rho, absolute_phases)
        histories.append(val)
        train_losses.append(float(np.mean(losses)))
        if val < best_loss:
            best_loss, best_round = val, r
            best_weights = copy.deepcopy(global_model.state_dict())
        if r in (1, 5):
            models[f"fed_{r}round"] = copy.deepcopy(global_model).cpu()
        print(f"[FL] round {r}/{config.FL_ROUNDS}: train={train_losses[-1]:.6f} "
              f"validation={val:.6f}", flush=True)
        if progress:
            progress({"round": r, "validation_loss": histories, "train_loss": train_losses})
        if r >= config.MIN_ROUNDS and plateau(histories, config.PATIENCE, config.MIN_DELTA):
            break
    rounds = len(histories)
    fed_s = time.perf_counter() - start
    global_model.load_state_dict(best_weights)
    models["fed_ris"] = global_model.cpu()

    # Final budget is actual work expended through the stopping round, including
    # rounds after the selected checkpoint. Do not charge just the best round.
    client_steps = rounds * per_round
    total_steps = len(td) * client_steps
    pooled = tuple(torch.cat([d[i] for d in td]) for i in range(4))
    central = copy.deepcopy(initial)
    central_rng = np.random.default_rng(config.SEED + 200)
    done, opt = 0, None
    central_history = []
    central_best, central_best_steps, central_best_weights = math.inf, 0, None
    central_selected = {}
    start = time.perf_counter()
    for target, key in ((client_steps, "centralized_client_budget"),
                        (total_steps, "centralized_dl")):
        while done < target:
            chunk = min(per_round * len(td), target - done)
            _, opt = train_steps(central, pooled, chunk, config.BATCH_SIZE, lr,
                                  central_rng, rho, absolute_phases, opt)
            done += chunk
            val = validation_loss(central, vd, rho, absolute_phases)
            central_history.append({"steps": done, "validation_loss": val})
            if val < central_best:
                central_best, central_best_steps = val, done
                central_best_weights = copy.deepcopy(central.state_dict())
        selected = copy.deepcopy(central)
        selected.load_state_dict(central_best_weights)
        models[key] = selected.cpu()
        central_selected[key] = central_best_steps
        print(f"[central] {key}: {done} updates, validation="
              f"{central_history[-1]['validation_loss']:.6f}", flush=True)
    central_s = time.perf_counter() - start

    local_models, local_validation, local_selected = [], [], []
    start = time.perf_counter()
    for t, (d, v) in enumerate(zip(td, vd)):
        model = copy.deepcopy(initial)
        local_rng = np.random.default_rng(config.SEED + 300 + t)
        opt, best, selected_steps, selected_weights = None, math.inf, 0, None
        for steps_done in range(per_round, client_steps + 1, per_round):
            _, opt = train_steps(model, d, per_round, config.BATCH_SIZE, lr,
                                 local_rng, rho, absolute_phases, opt)
            val = validation_loss(model, [v], rho, absolute_phases)
            if val < best:
                best, selected_steps = val, steps_done
                selected_weights = copy.deepcopy(model.state_dict())
        model.load_state_dict(selected_weights)
        local_validation.append(best)
        local_selected.append(selected_steps)
        local_models.append(model.cpu())
        print(f"[local] tile {t + 1}/{len(td)}: {client_steps} updates", flush=True)
    models["local_only"] = local_models
    parameters = sum(p.numel() for p in initial.parameters())
    # Current numerical training really uses float32 exchange. No unimplemented
    # INT8 compression discount. Count trainable state only; buffers are fixed.
    payload = parameters * 4
    traffic = 2 * len(td) * rounds * payload
    csi_bytes = len(td) * len(train[0]) * (train[0].num_ris_elements + 1) * 8
    stopping = "validation_plateau" if rounds < config.FL_ROUNDS or plateau(
        histories, config.PATIENCE, config.MIN_DELTA) else "budget_exhausted"
    meta = {
        "fl_rounds": rounds, "max_rounds": config.FL_ROUNDS,
        "local_epochs": config.LOCAL_EPOCHS, "steps_per_round_per_client": per_round,
        "fl_steps_per_client": client_steps, "fl_total_optimizer_steps": total_steps,
        "centralized_optimizer_steps": total_steps,
        "centralized_client_budget_steps": client_steps,
        "local_only_steps_per_client": client_steps,
        "batch_size": config.BATCH_SIZE, "learning_rate": lr,
        "optimizer": "Adam", "schedule": "constant",
        "fl_optimizer_state": "reset each round",
        "central_and_local_optimizer_state": "persistent",
        "objective": "local genie-normalized rate; not whole-aperture gradient",
        "best_round": best_round, "stopping_reason": stopping,
        "convergence_claim": False, "validation_patience": config.PATIENCE,
        "validation_min_delta": config.MIN_DELTA,
        "fl_round_losses": train_losses, "validation_losses": histories,
        "centralized_validation_history": central_history,
        "centralized_selected_steps": central_selected,
        "local_selected_steps": local_selected,
        "local_validation_losses": local_validation,
        "model_type": config.MODEL_TYPE, "model_parameters": parameters,
        "train_samples_per_tile": len(train[0]), "num_tiles": len(td),
        "communication_dtype": "float32", "payload_bytes_per_model": payload,
        "fl_total_communication_bytes": traffic, "raw_csi_upload_bytes": csi_bytes,
        "traffic_ratio_fl_to_raw_csi": traffic / csi_bytes,
        "crossover_samples_per_tile_analytic_only": 2 * rounds * payload /
                                                    ((train[0].num_ris_elements + 1) * 8),
        "local_mrc_training_bytes": 0, "local_only_training_bytes": 0,
        "training_byte_fields_scope": "model exchange; excludes channel acquisition and data delivery",
        "fixed_round_training_bytes": {str(r): 2 * len(td) * r * payload
                                       for r in (1, 5) if r <= rounds},
        "fl_wall_clock_s": fed_s, "centralized_wall_clock_s": central_s,
        "local_only_wall_clock_s": time.perf_counter() - start,
    }
    return models, meta

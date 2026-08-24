"""SVD-free first-step anchors plus a parity-structured FTBC tail."""

from collections import OrderedDict

import torch

from models.layer import SignedIF


def named_signed_layers(model):
    return OrderedDict(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, SignedIF)
    )


def full_bias_matrix(module):
    if module.ftbc_mode != "full" or module.time_based_bias is None:
        raise RuntimeError("Expected a calibrated Full-FTBC teacher layer")
    if len(module.time_based_bias) != int(module.T):
        raise RuntimeError("Full-FTBC teacher has an incomplete time schedule")
    return torch.stack(tuple(module.time_based_bias), dim=0).float()


def parity_anchor_basis(time_steps, *, device=None, dtype=torch.float32):
    """Return [first anchor, second anchor, tail mean, tail parity]."""
    time_steps = int(time_steps)
    if time_steps < 3:
        raise ValueError("Parity-Anchor FTBC requires at least three time steps")
    basis = torch.zeros(time_steps, 4, device=device, dtype=dtype)
    basis[0, 0] = 1
    basis[1, 1] = 1
    basis[2:, 2] = 1
    tail_times = torch.arange(2, time_steps, device=device)
    basis[2:, 3] = torch.where(
        tail_times.remainder(2) == 0,
        basis.new_ones(()),
        basis.new_full((), -1),
    )
    return basis


@torch.no_grad()
def compress_full_ftbc_teacher(model):
    """Replace Full-FTBC schedules with four fixed structured coefficients."""
    layers = named_signed_layers(model)
    if not layers:
        raise RuntimeError("No SignedIF layers found")
    time_steps = {int(module.T) for module in layers.values()}
    if len(time_steps) != 1:
        raise RuntimeError("All SignedIF layers must use the same time horizon")
    time_steps = time_steps.pop()
    if time_steps <= 4:
        raise ValueError("Compression is only defined above the Full fallback horizon")

    first_teacher = full_bias_matrix(next(iter(layers.values())))
    basis = parity_anchor_basis(
        time_steps,
        device=first_teacher.device,
        dtype=first_teacher.dtype,
    )
    projector = torch.linalg.pinv(basis)

    layer_reports = OrderedDict()
    total_error_energy = 0.0
    total_teacher_energy = 0.0
    total_channels = 0
    for name, module in layers.items():
        teacher = full_bias_matrix(module)
        coefficients = torch.matmul(projector, teacher).contiguous()
        reconstruction = torch.matmul(basis, coefficients)
        error = reconstruction - teacher
        denominator = teacher.square().mean().clamp_min(
            torch.finfo(teacher.dtype).eps
        )
        error_energy = error.square().sum()
        teacher_energy = teacher.square().sum()

        module.set_ftbc_mode("parity_anchor")
        module.parity_anchor_bias = coefficients
        total_error_energy += float(error_energy.detach().cpu().item())
        total_teacher_energy += float(teacher_energy.detach().cpu().item())
        total_channels += int(teacher.shape[1])
        layer_reports[name] = {
            "representation": "parity_anchor",
            "channels": int(teacher.shape[1]),
            "mse": float(error.square().mean().detach().cpu().item()),
            "nrmse": float(
                torch.sqrt(error.square().mean() / denominator).detach().cpu().item()
            ),
            "max_abs_error": float(error.abs().max().detach().cpu().item()),
        }

    explained_energy = 1.0 - total_error_energy / max(
        total_teacher_energy,
        torch.finfo(torch.float32).eps,
    )
    return {
        "coefficient_count": 4,
        "time_steps": time_steps,
        "compressed_channels": total_channels,
        "basis_stored": False,
        "threshold_normalize": False,
        "structure": "t0 anchor + t1 anchor + tail mean + tail parity",
        "explained_energy": float(max(min(explained_energy, 1.0), 0.0)),
        "layers": layer_reports,
    }

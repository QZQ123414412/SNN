"""Shared monotonic time-scale schedules for signed refinement coding."""

import torch


def make_time_scales(
    time_steps,
    mode="rate",
    ratio=2.0,
    device=None,
    dtype=None,
):
    """Return positive, non-increasing temporal scales with mean one."""
    time_steps = int(time_steps)
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")

    dtype = dtype or torch.float32
    if mode == "rate":
        return torch.ones(time_steps, device=device, dtype=dtype)
    if mode != "geometric":
        raise ValueError(f"Unsupported temporal scale mode: {mode}")

    ratio = float(ratio)
    if ratio < 1.0:
        raise ValueError("ratio must be greater than or equal to one")
    if time_steps == 1 or ratio == 1.0:
        return torch.ones(time_steps, device=device, dtype=dtype)

    exponents = -torch.arange(
        time_steps,
        device=device,
        dtype=torch.float64,
    )
    scales = torch.pow(torch.tensor(ratio, device=device, dtype=torch.float64), exponents)
    scales = scales * (time_steps / scales.sum())
    scales = scales.to(dtype=dtype)

    if not torch.isfinite(scales).all() or not torch.all(scales > 0):
        raise ValueError("temporal scales must be finite and positive")
    if not torch.all(scales[:-1] >= scales[1:]):
        raise ValueError("temporal scales must be non-increasing")
    return scales

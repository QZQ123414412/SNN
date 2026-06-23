"""Generate validated monotonic temporal weights for refinement coding."""

import torch


def _validate_time_steps(time_steps):
    time_steps = int(time_steps)
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    return time_steps


def _normalize_and_validate(weights):
    if weights.dim() != 1:
        raise ValueError("temporal weights must be one-dimensional")
    if not torch.isfinite(weights).all():
        raise ValueError("temporal weights must be finite")
    if not torch.all(weights > 0):
        raise ValueError("temporal weights must be positive")
    if weights.numel() > 1 and not torch.all(weights[:-1] >= weights[1:]):
        raise ValueError("temporal weights must be non-increasing")
    return weights / weights.sum()


def make_time_weights(
    time_steps,
    mode="uniform",
    ratio=2.0,
    custom_weights=None,
    device=None,
    dtype=None,
):
    """Return positive, non-increasing temporal weights whose sum is one."""
    time_steps = _validate_time_steps(time_steps)
    dtype = dtype or torch.float32

    if time_steps == 1:
        return torch.ones(1, device=device, dtype=dtype)

    if mode == "uniform":
        raw = torch.ones(time_steps, device=device, dtype=torch.float64)
    elif mode in {"binary", "geometric"}:
        effective_ratio = 2.0 if mode == "binary" else float(ratio)
        if effective_ratio < 1.0:
            raise ValueError("ratio must be greater than or equal to one")
        exponents = -torch.arange(
            time_steps,
            device=device,
            dtype=torch.float64,
        )
        base = torch.tensor(
            effective_ratio,
            device=device,
            dtype=torch.float64,
        )
        raw = torch.pow(base, exponents)
    elif mode == "custom":
        if custom_weights is None:
            raise ValueError("custom_weights are required for custom mode")
        raw = torch.as_tensor(
            custom_weights,
            device=device,
            dtype=torch.float64,
        )
        if raw.numel() != time_steps:
            raise ValueError("custom_weights length must equal time_steps")
    else:
        raise ValueError(f"Unsupported temporal weight mode: {mode}")

    return _normalize_and_validate(raw).to(dtype=dtype)


def make_event_scales(
    time_steps,
    mode="uniform",
    ratio=2.0,
    custom_weights=None,
    device=None,
    dtype=None,
):
    """Return `T*w_t`; multiplying events by these preserves mean decoding."""
    weights = make_time_weights(
        time_steps=time_steps,
        mode=mode,
        ratio=ratio,
        custom_weights=custom_weights,
        device=device,
        dtype=dtype,
    )
    return weights * int(time_steps)

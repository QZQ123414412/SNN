"""Teacher-compressed Temporal-LR FTBC and residual-gated SNM helpers."""

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


@torch.no_grad()
def compress_full_ftbc_teacher(
    model,
    rank,
    full_layer_names=(),
    threshold_normalize=True,
):
    """Compress all non-exempt Full-FTBC schedules with one shared SVD basis."""
    rank = int(rank)
    if rank <= 0:
        raise ValueError("Temporal-LR rank must be positive")

    layers = named_signed_layers(model)
    if not layers:
        raise RuntimeError("No SignedIF layers found")
    full_layer_names = set(full_layer_names)
    unknown = full_layer_names - set(layers)
    if unknown:
        raise ValueError(f"Unknown Full-FTBC exemption layers: {sorted(unknown)}")

    compressed_names = [name for name in layers if name not in full_layer_names]
    if not compressed_names:
        raise ValueError("At least one layer must use Temporal-LR")

    normalized_matrices = []
    teacher_matrices = {}
    thresholds = {}
    for name, module in layers.items():
        teacher = full_bias_matrix(module)
        teacher_matrices[name] = teacher
        threshold = float(module.thresh.detach().abs().item())
        threshold = max(threshold, torch.finfo(torch.float32).eps)
        thresholds[name] = threshold
        if name in compressed_names:
            normalized_matrices.append(
                teacher / threshold if threshold_normalize else teacher
            )

    stacked = torch.cat(normalized_matrices, dim=1)
    effective_rank = min(rank, stacked.shape[0], stacked.shape[1])
    basis, singular_values, _ = torch.linalg.svd(stacked, full_matrices=False)
    basis = basis[:, :effective_rank].contiguous()
    total_energy = singular_values.square().sum()
    retained_energy = singular_values[:effective_rank].square().sum()
    explained_energy = float(
        (retained_energy / total_energy.clamp_min(torch.finfo(total_energy.dtype).eps))
        .detach()
        .cpu()
        .item()
    )

    layer_reports = OrderedDict()
    basis_owner_assigned = False
    for name, module in layers.items():
        teacher = teacher_matrices[name]
        if name in full_layer_names:
            reconstruction = teacher
            representation = "full"
        else:
            source = teacher / thresholds[name] if threshold_normalize else teacher
            coefficient = torch.matmul(basis.transpose(0, 1), source)
            if threshold_normalize:
                coefficient = coefficient * thresholds[name]
            reconstruction = torch.matmul(basis, coefficient)

            module.set_ftbc_mode("temporal_low_rank")
            module.temporal_basis = basis
            module.temporal_coeff = coefficient.contiguous()
            module.owns_temporal_basis = not basis_owner_assigned
            basis_owner_assigned = True
            representation = "temporal_low_rank"

        error = reconstruction - teacher
        denominator = teacher.square().mean().clamp_min(
            torch.finfo(teacher.dtype).eps
        )
        layer_reports[name] = {
            "representation": representation,
            "channels": int(teacher.shape[1]),
            "mse": float(error.square().mean().detach().cpu().item()),
            "nrmse": float(
                torch.sqrt(error.square().mean() / denominator).detach().cpu().item()
            ),
            "max_abs_error": float(error.abs().max().detach().cpu().item()),
        }

    return {
        "requested_rank": rank,
        "effective_rank": effective_rank,
        "time_steps": int(stacked.shape[0]),
        "compressed_channels": int(stacked.shape[1]),
        "full_layer_names": sorted(full_layer_names),
        "threshold_normalize": bool(threshold_normalize),
        "explained_energy": explained_energy,
        "layers": layer_reports,
    }


def gate_groups(model, architecture):
    """Return four disjoint architecture-aware SNM gate groups."""
    names = tuple(named_signed_layers(model))
    architecture = architecture.lower()
    if architecture == "resnet20":
        final_name = "conv4_x.2.act"
        groups = OrderedDict(
            [
                (
                    "early",
                    [name for name in names if name == "conv1.2" or name.startswith("conv2_x")],
                ),
                ("middle", [name for name in names if name.startswith("conv3_x")]),
                (
                    "late",
                    [
                        name
                        for name in names
                        if name.startswith("conv4_x") and name != final_name
                    ],
                ),
                ("final", [final_name]),
            ]
        )
    elif architecture == "vgg16":
        groups = OrderedDict(
            [
                ("early", [name for name in names if name.startswith(("layer1", "layer2"))]),
                ("middle", [name for name in names if name.startswith("layer3")]),
                ("late", [name for name in names if name.startswith(("layer4", "layer5"))]),
                ("final", [name for name in names if name.startswith("classifier")]),
            ]
        )
    else:
        raise ValueError(f"Unsupported gate architecture: {architecture}")

    flattened = [name for group in groups.values() for name in group]
    if len(flattened) != len(names) or set(flattened) != set(names):
        raise RuntimeError(
            f"SNM gate groups do not partition {architecture}: "
            f"missing={sorted(set(names) - set(flattened))}, "
            f"duplicates={len(flattened) - len(set(flattened))}"
        )
    return groups


def set_group_margins(model, architecture, margins):
    layers = named_signed_layers(model)
    groups = gate_groups(model, architecture)
    if set(margins) != set(groups):
        raise ValueError(
            f"Margins must contain exactly these groups: {list(groups)}"
        )
    for group_name, layer_names in groups.items():
        margin = float(margins[group_name])
        for name in layer_names:
            layers[name].set_snm_negative_margin(margin)
    return sum(len(layer_names) > 0 for layer_names in groups.values())


def snm_runtime_state_bytes_per_sample(model):
    """Report R0's materialized membrane plus transmitted-credit state.

    Gated-SNM reuses ``transmitted`` and therefore adds no dense per-sample
    state beyond R0; its only extra storage is the small set of scalar margins.
    """
    total = 0
    for module in named_signed_layers(model).values():
        for state in (module.mem, module.transmitted):
            if not torch.is_tensor(state) or state.dim() == 0:
                continue
            batch = max(int(state.shape[0]), 1)
            total += state.numel() * state.element_size() // batch
    return int(total)

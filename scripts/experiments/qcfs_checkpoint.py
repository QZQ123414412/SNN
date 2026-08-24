"""Strict loading and provenance helpers for QCFS checkpoints."""

import hashlib
from pathlib import Path

import torch

from models import SignedIF, modelpool


def checkpoint_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must contain a state-dict mapping")
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break
    if not checkpoint or not all(isinstance(key, str) for key in checkpoint):
        raise RuntimeError("Checkpoint does not contain a valid state dict")
    if all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def _normalize_legacy_threshold_keys(state_dict):
    """Return canonical QCFS keys without weakening strict model loading.

    Early QCFS checkpoints stored each activation threshold as ``.up``.  The
    current IF/SignedIF modules call the same tensor ``.thresh``.  Translate
    only that exact suffix and reject collisions so a malformed checkpoint
    cannot silently choose one of two threshold tensors.
    """
    normalized = {}
    legacy_threshold_keys = 0
    for key, value in state_dict.items():
        canonical_key = key[:-3] + ".thresh" if key.endswith(".up") else key
        if canonical_key in normalized:
            raise RuntimeError(
                "Checkpoint contains colliding threshold keys after legacy "
                f"normalization: {key!r} -> {canonical_key!r}"
            )
        normalized[canonical_key] = value
        legacy_threshold_keys += int(canonical_key != key)
    return normalized, legacy_threshold_keys


def _normalize_legacy_module_names(state_dict, architecture):
    """Translate only known architecture-local legacy module names."""
    normalized = {}
    renamed = 0
    for key, value in state_dict.items():
        canonical_key = key
        if architecture == "resnet34":
            canonical_key = canonical_key.replace(".relu.", ".act.")
        if canonical_key in normalized:
            raise RuntimeError(
                "Checkpoint contains colliding module keys after legacy "
                f"normalization: {key!r} -> {canonical_key!r}"
            )
        normalized[canonical_key] = value
        renamed += int(canonical_key != key)
    return normalized, renamed


def load_qcfs_pair(checkpoint_path, dataset, architecture, device):
    """Load one QCFS checkpoint into exact IF and SignedIF architectures."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"QCFS checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict, legacy_threshold_keys = _normalize_legacy_threshold_keys(
        _extract_state_dict(checkpoint)
    )
    state_dict, legacy_module_keys = _normalize_legacy_module_names(
        state_dict, architecture
    )

    ann = modelpool(architecture, dataset)
    try:
        ann.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint is not an exact {dataset}/{architecture} QCFS model: "
            f"{error}"
        ) from error

    signed_architecture = f"{architecture}_signed"
    snn = modelpool(signed_architecture, dataset)
    signed_state = dict(state_dict)
    threshold_names = []
    for name, module in snn.named_modules():
        if isinstance(module, SignedIF):
            threshold_key = f"{name}.thresh"
            if threshold_key not in state_dict:
                raise RuntimeError(
                    f"QCFS threshold missing for activation {name!r}: "
                    f"expected key {threshold_key!r}"
                )
            negative_key = f"{name}.neg_thresh"
            signed_state[negative_key] = -state_dict[threshold_key]
            threshold_names.append(name)
    if not threshold_names:
        raise RuntimeError(
            f"Signed architecture {signed_architecture!r} has no SignedIF layers"
        )
    try:
        snn.load_state_dict(signed_state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "QCFS-to-SignedIF conversion was not exact: " + str(error)
        ) from error

    ann.to(device)
    snn.to(device)
    metadata = {
        "path": str(checkpoint_path),
        "filename": checkpoint_path.name,
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": checkpoint_sha256(checkpoint_path),
        "dataset": dataset,
        "architecture": architecture,
        "qcfs_layers": len(threshold_names),
        "legacy_threshold_keys_normalized": legacy_threshold_keys,
        "legacy_module_keys_normalized": legacy_module_keys,
    }
    return ann, snn, metadata

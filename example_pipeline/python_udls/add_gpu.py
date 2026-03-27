"""
Add two same-shaped matrices on GPU using torch + DLPack.

Supports both call styles:
- Direct invoke: invoke(a_capsule, b_capsule) -> float
- Batched queue invoke: invoke([(a_capsule, b_capsule), ...]) -> list[float]
"""

from __future__ import annotations


def _import_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"Failed to import torch for add_gpu: {exc}") from exc
    return torch


def _add_and_reduce(a_capsule, b_capsule) -> float:
    torch = _import_torch()
    a = torch.utils.dlpack.from_dlpack(a_capsule)
    b = torch.utils.dlpack.from_dlpack(b_capsule)

    print(a.shape)
    print(a.device)
    if not a.is_cuda or not b.is_cuda:
        raise RuntimeError("Expected CUDA tensors from DLPack capsules.")
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")

    c = a + b
    return float(c.sum().item())


def invoke(arg0, arg1=None):
    # Direct path: invoke(a_capsule, b_capsule)
    if arg1 is not None:
        return _add_and_reduce(arg0, arg1)

    # Batched path: invoke(items) where items is iterable of (a_capsule, b_capsule)
    items = list(arg0)
    if not items:
        return []

    return [_add_and_reduce(a_cap, b_cap) for a_cap, b_cap in items]

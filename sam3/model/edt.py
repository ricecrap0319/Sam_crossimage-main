# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Euclidean Distance Transform — Triton GPU kernel with OpenCV CPU fallback for Windows."""

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# ---------------------------------------------------------------------------
# Triton kernel (Linux / Mac with triton installed)
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:
    @triton.jit
    def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width

        k = 0
        for q in range(1, length):
            cur_input = tl.load(inputs_ptr + block_start + (q * stride))
            r = tl.load(v + block_start + (k * stride))
            z_k = tl.load(z + block_start + (k * stride))
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            while s <= z_k and k - 1 >= 0:
                k = k - 1
                r = tl.load(v + block_start + (k * stride))
                z_k = tl.load(z + block_start + (k * stride))
                previous_input = tl.load(inputs_ptr + block_start + (r * stride))
                s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            k = k + 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            if k + 1 < length:
                tl.store(z + block_start + ((k + 1) * stride), 1e9)

        k = 0
        for q in range(length):
            while (
                k + 1 < length
                and tl.load(
                    z + block_start + ((k + 1) * stride), mask=(k + 1) < length, other=q
                )
                < q
            ):
                k += 1
            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)

    def _edt_triton_impl(data: torch.Tensor) -> torch.Tensor:
        assert data.dim() == 3
        assert data.is_cuda
        B, H, W = data.shape
        data = data.contiguous()
        output = torch.where(data, 1e18, 0.0)
        assert output.is_contiguous()
        parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
        parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18
        edt_kernel[(B, H)](output.clone(), output, parabola_loc, parabola_inter, H, W, horizontal=True)
        parabola_loc.zero_()
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18
        edt_kernel[(B, W)](output.clone(), output, parabola_loc, parabola_inter, H, W, horizontal=False)
        return output.sqrt()

# ---------------------------------------------------------------------------
# OpenCV fallback (Windows / no triton)
# ---------------------------------------------------------------------------
def _edt_cv2_fallback(data: torch.Tensor) -> torch.Tensor:
    """Pure-Python EDT using cv2.distanceTransform — works on CPU and CUDA tensors."""
    import cv2
    import numpy as np

    device = data.device
    data_cpu = data.cpu().numpy().astype(np.uint8)   # (B, H, W)
    B = data_cpu.shape[0]
    results = []
    for b in range(B):
        # distanceTransform: distance to nearest ZERO pixel
        # data==1 means foreground (non-zero), we want distance to nearest zero
        src = (1 - data_cpu[b]).astype(np.uint8)     # invert: zeros become ones
        dt = cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        results.append(dt)
    out = torch.from_numpy(np.stack(results, axis=0)).to(device=device, dtype=torch.float32)
    return out


# ---------------------------------------------------------------------------
# Public API — automatically picks best implementation
# ---------------------------------------------------------------------------
def edt_triton(data: torch.Tensor) -> torch.Tensor:
    """
    Euclidean Distance Transform of a batch of binary images (B, H, W).
    Uses the Triton GPU kernel when available, falls back to cv2 on Windows.
    """
    if _TRITON_AVAILABLE and data.is_cuda:
        return _edt_triton_impl(data)
    return _edt_cv2_fallback(data)

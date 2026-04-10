"""Mamba-3 SISO in pure JAX. No Pallas needed.

Replaces ~6000 lines of Triton forward/backward/angle_dt kernels with ~200 lines.
Backward pass is handled automatically by jax.grad through jax.lax.scan.
"""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

TWO_PI = 2 * math.pi


# ---------------------------------------------------------------------------
# angle_dt cumsum
# ---------------------------------------------------------------------------

def _angle_dt_scan_body(state, chunk):
    """state: (BH, dim), chunk: (BH, chunk_size, dim)"""
    chunk_cumsum = jnp.cumsum(chunk, axis=1)
    out = chunk_cumsum + state[:, None, :]
    out = out - TWO_PI * jnp.floor(out / TWO_PI)
    new_state = state + jnp.sum(chunk, axis=1)
    new_state = new_state - TWO_PI * jnp.floor(new_state / TWO_PI)
    return new_state, out


def angle_dt_cumsum(
    angles: jnp.ndarray,
    dt: jnp.ndarray,
    chunk_size: int = 64,
) -> jnp.ndarray:
    """Compute cumsum(tanh(angles) * pi * dt) mod 2pi, chunked.

    Args:
        angles:     (batch, seqlen, nheads, dim)
        dt:         (batch, nheads, seqlen)
    Returns:
        out:         (batch, seqlen, nheads, dim)
    """
    batch, seqlen, nheads, dim = angles.shape
    nchunks = seqlen // chunk_size
    BH = batch * nheads

    vals = jnp.tanh(angles.astype(jnp.float32)) * math.pi
    # (batch, seqlen, nheads, dim) -> (batch, nheads, nchunks, chunk_size, dim)
    vals = jnp.transpose(
        vals.reshape(batch, nchunks, chunk_size, nheads, dim), (0, 3, 1, 2, 4)
    )
    dt_r = dt.reshape(batch, nheads, nchunks, chunk_size)
    vals = vals * dt_r[..., None]
    # -> (BH, nchunks, chunk_size, dim)
    vals = vals.reshape(BH, nchunks, chunk_size, dim)
    # scan axis is nchunks, so transpose: (nchunks, BH, chunk_size, dim)
    vals = jnp.transpose(vals, (1, 0, 2, 3))

    init_state = jnp.zeros((BH, dim), dtype=jnp.float32)

    _, out = lax.scan(_angle_dt_scan_body, init_state, vals)
    # out: (nchunks, BH, chunk_size, dim) -> (batch, nheads, nchunks, chunk_size, dim)
    out = jnp.transpose(out, (1, 0, 2, 3)).reshape(batch, nheads, nchunks, chunk_size, dim)
    # -> (batch, seqlen, nheads, dim)
    out = jnp.transpose(out, (0, 2, 3, 1, 4)).reshape(batch, seqlen, nheads, dim)
    return out


# ---------------------------------------------------------------------------
# Rotary embedding (batched, no vmap)
# ---------------------------------------------------------------------------

def _apply_rotary_batched(x, cos, sin):
    """Apply rotary embedding, batched.

    x:   (BH, chunk_size, headdim_qk)
    cos: (BH, chunk_size, n_rot)
    sin: (BH, chunk_size, n_rot)
    """
    BH, L, headdim = x.shape
    n_rot = cos.shape[-1]
    pairs = x.reshape(BH, L, headdim // 2, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]

    ro0 = x0[..., :n_rot] * cos - x1[..., :n_rot] * sin
    ro1 = x0[..., :n_rot] * sin + x1[..., :n_rot] * cos

    out0 = jnp.concatenate([ro0, x0[..., n_rot:]], axis=-1)
    out1 = jnp.concatenate([ro1, x1[..., n_rot:]], axis=-1)
    return jnp.stack([out0, out1], axis=-1).reshape(BH, L, headdim)


# ---------------------------------------------------------------------------
# Scale / gamma from DT and Trap
# ---------------------------------------------------------------------------

def _compute_scale_gamma(DT, Trap):
    """DT, Trap: (batch, nheads, seqlen). Returns scale, gamma same shape."""
    dt_f32 = DT.astype(jnp.float32)
    trap_sig = jax.nn.sigmoid(Trap.astype(jnp.float32))
    dt_shifted = jnp.concatenate(
        [dt_f32[:, :, 1:], jnp.zeros_like(dt_f32[:, :, :1])], axis=-1
    )
    trap_sig_shifted = jnp.concatenate(
        [trap_sig[:, :, 1:], jnp.zeros_like(trap_sig[:, :, :1])], axis=-1
    )
    gamma = dt_f32 * trap_sig
    scale = dt_shifted * (1 - trap_sig_shifted) + gamma
    return scale, gamma


# ---------------------------------------------------------------------------
# Core: single flat scan with batched matmuls, no vmap
# ---------------------------------------------------------------------------

def _mamba3_scan(
    Q, K, V, ADT, Q_bias, K_bias,
    angles_cos, angles_sin,
    scale, gamma, D, Z,
    chunk_size,
):
    """Chunked SSM. All preprocessing done per-chunk inside scan body.

    Q, K:           (batch, seqlen, nheads_qk, headdim_qk)
    V:              (batch, seqlen, nheads, headdim_v)
    ADT:            (batch, nheads, seqlen)
    Q_bias, K_bias: (nheads, headdim_qk)
    angles_cos/sin: (batch, seqlen, nheads, headdim_angles) — precomputed
    scale, gamma:   (batch, nheads, seqlen)
    D:              (nheads,)
    Z:              (batch, seqlen, nheads, headdim_v) or None
    """
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    nheads = V.shape[2]
    headdim_v = V.shape[3]
    nchunks = seqlen // chunk_size
    BH = batch * nheads
    gqa_ratio = nheads // nheads_qk

    def _to_scan(x_blhd, nh=nheads):
        """(batch, seqlen, nh, dim) -> (nchunks, batch*nh, chunk_size, dim)"""
        x = x_blhd.reshape(batch, nchunks, chunk_size, nh, -1)
        x = jnp.transpose(x, (1, 0, 3, 2, 4))
        return x.reshape(nchunks, batch * nh, chunk_size, -1)

    def _to_scan_bhl(x):
        """(batch, nheads, seqlen) -> (nchunks, BH, chunk_size)"""
        return jnp.transpose(
            x.reshape(batch, nheads, nchunks, chunk_size), (2, 0, 1, 3)
        ).reshape(nchunks, BH, chunk_size)

    v_s = _to_scan(V)
    adt_s = _to_scan_bhl(ADT)
    scale_s = _to_scan_bhl(scale)
    gamma_s = _to_scan_bhl(gamma)
    cos_s = _to_scan(angles_cos)
    sin_s = _to_scan(angles_sin)

    # GQA expand Q, K before scan: (nchunks, BHq, ...) -> (nchunks, BH, ...)
    q_s = _to_scan(Q, nheads_qk)
    k_s = _to_scan(K, nheads_qk)
    if gqa_ratio > 1:
        q_s = jnp.repeat(q_s, gqa_ratio, axis=1)
        k_s = jnp.repeat(k_s, gqa_ratio, axis=1)

    q_bias_bh = jnp.broadcast_to(Q_bias, (batch, nheads, headdim_qk)).reshape(BH, headdim_qk)
    k_bias_bh = jnp.broadcast_to(K_bias, (batch, nheads, headdim_qk)).reshape(BH, headdim_qk)
    d_bh = jnp.broadcast_to(D, (batch, nheads)).reshape(BH)

    state_init = jnp.zeros((BH, headdim_v, headdim_qk), dtype=jnp.float32)
    causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=-1)

    def _scan_body(ssm_state, inputs):
        """One chunk, all batch*heads in parallel via batched matmul.

        ssm_state: (BH, headdim_v, headdim_qk)
        """
        q, k, v, adt, sc, gm, cos_a, sin_a = inputs
        # q, k: (BH, chunk_size, hqk)
        # v:    (BH, chunk_size, hv)
        # adt, sc, gm: (BH, chunk_size)
        # cos_a, sin_a: (BH, chunk_size, ha)

        # Bias
        q = q + q_bias_bh[:, None, :]
        k = k + k_bias_bh[:, None, :]

        # QK dot (before rotary)
        qk_dot = jnp.sum(q * k, axis=-1) * gm  # (BH, chunk_size)

        # Rotary (precomputed cos/sin)
        q = _apply_rotary_batched(q, cos_a, sin_a)
        k = _apply_rotary_batched(k, cos_a, sin_a)

        # Scale K
        k = k * sc[:, :, None]

        # --- SSM ---
        da_cs = jnp.cumsum(adt, axis=-1)
        da_cs_last = da_cs[:, -1]

        # Inter-chunk: (BH, L, hqk) @ (BH, hqk, hv) -> (BH, L, hv)
        acc_o = jnp.matmul(q, jnp.transpose(ssm_state, (0, 2, 1)))
        acc_o = acc_o * jnp.exp(da_cs)[:, :, None]

        # Intra-chunk: (BH, L, hqk) @ (BH, hqk, L) -> (BH, L, L)
        s = jnp.matmul(q, jnp.transpose(k, (0, 2, 1)))
        s = s * jnp.exp(jnp.minimum(da_cs[:, :, None] - da_cs[:, None, :], 0.0))
        s = jnp.where(causal_mask[None, :, :], s, 0.0)
        acc_o = acc_o + jnp.matmul(s, v)

        # D-skip + QK diagonal
        acc_o = acc_o + (d_bh[:, None] + qk_dot)[:, :, None] * v

        # State update
        da_cs_rev = da_cs_last[:, None] - da_cs
        v_scaled = v * jnp.exp(da_cs_rev)[:, :, None]
        # (BH, hv, L) @ (BH, L, hqk) -> (BH, hv, hqk)
        new_state = (ssm_state * jnp.exp(da_cs_last)[:, None, None]
                     + jnp.matmul(jnp.transpose(v_scaled, (0, 2, 1)), k))

        return new_state, acc_o

    _scan_body_remat = jax.checkpoint(_scan_body)

    _, out_s = lax.scan(
        _scan_body_remat, state_init,
        (q_s, k_s, v_s, adt_s, scale_s, gamma_s, cos_s, sin_s),
    )
    # out_s: (nchunks, BH, chunk_size, hv) -> (batch, seqlen, nheads, hv)
    out = out_s.reshape(nchunks, batch, nheads, chunk_size, headdim_v)
    out = jnp.transpose(out, (1, 0, 3, 2, 4)).reshape(batch, seqlen, nheads, headdim_v)

    if Z is not None:
        out = out * jax.nn.silu(Z.astype(jnp.float32))

    return out


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def mamba3_siso_combined(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    ADT: jnp.ndarray,
    DT: jnp.ndarray,
    Trap: jnp.ndarray,
    Q_bias: jnp.ndarray,
    K_bias: jnp.ndarray,
    Angles: jnp.ndarray,
    D: Optional[jnp.ndarray] = None,
    Z: Optional[jnp.ndarray] = None,
    chunk_size: int = 64,
) -> jnp.ndarray:
    """Mamba-3 SISO forward pass in pure JAX.

    Args:
        Q:         (batch, seqlen, nheads_qk, headdim_qk)
        K:         (batch, seqlen, nheads_qk, headdim_qk)
        V:         (batch, seqlen, nheads, headdim_v)
        ADT:       (batch, nheads, seqlen) -- A*dt decay factor
        DT:        (batch, nheads, seqlen) -- time delta (post-softplus)
        Trap:      (batch, nheads, seqlen) -- trapezoidal factor (raw, pre-sigmoid)
        Q_bias:    (nheads, headdim_qk)
        K_bias:    (nheads, headdim_qk)
        Angles:    (batch, seqlen, nheads, headdim_angles)
        D:         (nheads,) or None -- skip connection
        Z:         (batch, seqlen, nheads, headdim_v) or None -- gating
        chunk_size: chunk size (default 64). Must be passed via functools.partial for jit.

    Returns:
        out: (batch, seqlen, nheads, headdim_v)
    """
    batch, seqlen, nheads, headdim_v = V.shape

    # Pad seqlen to multiple of chunk_size
    remainder = seqlen % chunk_size
    if remainder != 0:
        pad_len = chunk_size - remainder
        def _pad_seq(x, ax=1):
            pad_widths = [(0, 0)] * x.ndim
            pad_widths[ax] = (0, pad_len)
            return jnp.pad(x, pad_widths)
        Q = _pad_seq(Q)
        K = _pad_seq(K)
        V = _pad_seq(V)
        Angles = _pad_seq(Angles)
        ADT = _pad_seq(ADT, ax=2)
        DT = _pad_seq(DT, ax=2)
        Trap = _pad_seq(Trap, ax=2)
        if Z is not None:
            Z = _pad_seq(Z)

    # Cast to bf16 for compute (matching Triton kernel)
    Q = Q.astype(jnp.bfloat16)
    K = K.astype(jnp.bfloat16)
    V = V.astype(jnp.bfloat16)
    Angles = Angles.astype(jnp.bfloat16)
    if Z is not None:
        Z = Z.astype(jnp.bfloat16)

    # 1. Angle-DT cumsum
    angles_cumsum = angle_dt_cumsum(Angles, DT, chunk_size=chunk_size)

    # 2. Precompute cos/sin of angles (avoids trig in scan body)
    angles_cos = jnp.cos(angles_cumsum.astype(jnp.float32))
    angles_sin = jnp.sin(angles_cumsum.astype(jnp.float32))

    # 3. Scale/gamma
    scale, gamma = _compute_scale_gamma(DT, Trap)

    d_val = D if D is not None else jnp.zeros(nheads, dtype=jnp.float32)

    # 4. Core scan
    out = _mamba3_scan(
        Q, K, V, ADT, Q_bias, K_bias,
        angles_cos, angles_sin,
        scale, gamma, d_val, Z,
        chunk_size,
    )

    # Trim padding
    if remainder != 0:
        out = out[:, :seqlen]

    return out

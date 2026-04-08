"""Mamba-3 SISO in pure JAX. No Pallas needed.

Replaces ~6000 lines of Triton forward/backward/angle_dt kernels with ~200 lines.
Backward pass is handled automatically by jax.grad through jax.lax.scan.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax

LOG2E = math.log2(math.e)
TWO_PI = 2 * math.pi


# ---------------------------------------------------------------------------
# angle_dt cumsum
# ---------------------------------------------------------------------------

def _angle_dt_scan_body(state, chunk):
    """state: (dim,), chunk: (chunk_size, dim)"""
    chunk_cumsum = jnp.cumsum(chunk, axis=0)
    out = chunk_cumsum + state[None, :]
    out = out - TWO_PI * jnp.floor(out / TWO_PI)
    new_state = state + jnp.sum(chunk, axis=0)
    new_state = new_state - TWO_PI * jnp.floor(new_state / TWO_PI)
    return new_state, out


def angle_dt_cumsum(
    angles: jnp.ndarray,
    dt: jnp.ndarray,
    init_state: Optional[jnp.ndarray] = None,
    chunk_size: int = 64,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute cumsum(tanh(angles) * pi * dt) mod 2pi, chunked.

    Args:
        angles:     (batch, seqlen, nheads, dim)
        dt:         (batch, nheads, seqlen)
        init_state: (batch, nheads, dim) or None
    Returns:
        out:         (batch, seqlen, nheads, dim)
        final_state: (batch, nheads, dim)
    """
    batch, seqlen, nheads, dim = angles.shape
    nchunks = seqlen // chunk_size

    vals = jnp.tanh(angles.astype(jnp.float32)) * math.pi
    dt_r = dt.reshape(batch, nheads, nchunks, chunk_size)
    vals = jnp.transpose(
        vals.reshape(batch, nchunks, chunk_size, nheads, dim), (0, 3, 1, 2, 4)
    )
    vals = vals * dt_r[..., None]

    if init_state is None:
        init_state = jnp.zeros((batch, nheads, dim), dtype=jnp.float32)

    def _per_bh(state, chunks):
        return lax.scan(_angle_dt_scan_body, state, chunks)

    final_state, out = jax.vmap(jax.vmap(_per_bh))(init_state, vals)
    out = jnp.transpose(out, (0, 2, 3, 1, 4)).reshape(batch, seqlen, nheads, dim)
    return out, final_state


# ---------------------------------------------------------------------------
# Rotary embedding helper (chunk-level, no materialization of full sequence)
# ---------------------------------------------------------------------------

def _apply_rotary(x, cos, sin):
    """Apply rotary embedding. cos/sin cover only the rotated pairs.

    x:   (chunk_size, headdim_qk)
    cos: (chunk_size, n_rot)
    sin: (chunk_size, n_rot)
    """
    headdim = x.shape[-1]
    n_rot = cos.shape[-1]
    pairs = x.reshape(-1, headdim // 2, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]

    ro0 = x0[..., :n_rot] * cos - x1[..., :n_rot] * sin
    ro1 = x0[..., :n_rot] * sin + x1[..., :n_rot] * cos

    out0 = jnp.concatenate([ro0, x0[..., n_rot:]], axis=-1)
    out1 = jnp.concatenate([ro1, x1[..., n_rot:]], axis=-1)
    return jnp.stack([out0, out1], axis=-1).reshape(-1, headdim)


# ---------------------------------------------------------------------------
# Scale / gamma from DT and Trap (full-sequence, cheap)
# ---------------------------------------------------------------------------

def _compute_scale_gamma(DT, Trap):
    """Compute scale and gamma. DT, Trap: (batch, nheads, seqlen).
    Returns scale, gamma each (batch, nheads, seqlen)."""
    dt_f32 = DT.astype(jnp.float32)
    trap_sig = jax.nn.sigmoid(Trap.astype(jnp.float32))
    dt_shifted = jnp.concatenate(
        [dt_f32[:, :, 1:], jnp.zeros_like(dt_f32[:, :, :1])], axis=-1
    )
    trap_sig_shifted = jnp.concatenate(
        [jax.nn.sigmoid(Trap[:, :, 1:].astype(jnp.float32)),
         jnp.zeros_like(trap_sig[:, :, :1])], axis=-1
    )
    gamma = dt_f32 * trap_sig
    scale = dt_shifted * (1 - trap_sig_shifted) + gamma
    return scale, gamma


# ---------------------------------------------------------------------------
# Core: fused scan that does preprocessing inside the body
# ---------------------------------------------------------------------------

def _mamba3_scan(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    ADT: jnp.ndarray,
    DT: jnp.ndarray,
    Trap: jnp.ndarray,
    Q_bias: jnp.ndarray,
    K_bias: jnp.ndarray,
    angles_cumsum: jnp.ndarray,
    scale: jnp.ndarray,
    gamma: jnp.ndarray,
    D: jnp.ndarray,
    Z: Optional[jnp.ndarray],
    init_ssm_state: jnp.ndarray,
    chunk_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run the chunked SSM with preprocessing fused into the scan body.

    Q, K:           (batch, seqlen, nheads_qk, headdim_qk)  — NOT expanded
    V:              (batch, seqlen, nheads, headdim_v)
    ADT:            (batch, nheads, seqlen)
    DT:             (batch, nheads, seqlen)
    Trap:           (batch, nheads, seqlen)
    Q_bias, K_bias: (nheads, headdim_qk)
    angles_cumsum:  (batch, seqlen, nheads, headdim_angles)
    scale, gamma:   (batch, nheads, seqlen)
    D:              (nheads,)
    Z:              (batch, seqlen, nheads, headdim_v) or None
    init_ssm_state: (batch, nheads, headdim_v, headdim_qk)
    """
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    nheads = V.shape[2]
    headdim_v = V.shape[3]
    headdim_angles = angles_cumsum.shape[3]
    nchunks = seqlen // chunk_size
    gqa_ratio = nheads // nheads_qk

    # Chunk the per-head tensors: (batch, nheads, nchunks, chunk_size, ...)
    def _chunk_bhl(x):
        """(batch, nheads, seqlen) -> (batch, nheads, nchunks, chunk_size)"""
        return x.reshape(batch, nheads, nchunks, chunk_size)

    def _chunk_blhd(x):
        """(batch, seqlen, nheads, dim) -> (batch, nheads, nchunks, chunk_size, dim)"""
        return jnp.transpose(
            x.reshape(batch, nchunks, chunk_size, x.shape[2], x.shape[3]),
            (0, 3, 1, 2, 4),
        )

    v_c = _chunk_blhd(V)                      # (B, nheads, C, L, hv)
    adt_c = _chunk_bhl(ADT)                    # (B, nheads, C, L)
    scale_c = _chunk_bhl(scale)                # (B, nheads, C, L)
    gamma_c = _chunk_bhl(gamma)                # (B, nheads, C, L)
    ang_c = _chunk_blhd(angles_cumsum)         # (B, nheads, C, L, ha)

    # Q, K stay at nheads_qk: (batch, nheads_qk, nchunks, chunk_size, headdim_qk)
    q_c = _chunk_blhd(Q)
    k_c = _chunk_blhd(K)

    if Z is not None:
        z_c = _chunk_blhd(Z)                   # (B, nheads, C, L, hv)

    causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=-1)

    def _scan_body(ssm_state, inputs):
        """One chunk for a single (batch-elem, head).

        ssm_state: (headdim_v, headdim_qk)
        q, k:      (chunk_size, headdim_qk)  — shared across GQA group
        v:         (chunk_size, headdim_v)
        adt:       (chunk_size,)
        scale_ch:  (chunk_size,)
        gamma_ch:  (chunk_size,)
        ang:       (chunk_size, headdim_angles)
        q_bias_h:  (headdim_qk,)
        k_bias_h:  (headdim_qk,)
        d:         scalar
        """
        q, k, v, adt, scale_ch, gamma_ch, ang, q_bias_h, k_bias_h, d = inputs

        # --- Preprocessing (fused into scan body, per-chunk) ---
        # Bias
        q = q + q_bias_h
        k = k + k_bias_h

        # QK dot (before rotary)
        qk_dot = jnp.sum(q * k, axis=-1) * gamma_ch  # (chunk_size,)

        # Rotary
        cos_a = jnp.cos(ang.astype(jnp.float32))
        sin_a = jnp.sin(ang.astype(jnp.float32))
        q = _apply_rotary(q, cos_a, sin_a)
        k = _apply_rotary(k, cos_a, sin_a)

        # Scale K
        k = k * scale_ch[:, None]

        # --- SSM computation ---
        da = adt * LOG2E
        da_cs = jnp.cumsum(da)
        da_cs_last = jnp.sum(da)

        # Inter-chunk: Q @ State^T * exp2(da_cs)
        acc_o = (q @ ssm_state.T) * jnp.exp2(da_cs)[:, None]

        # Intra-chunk: causal(Q @ K^T * exp2(decay)) @ V
        s = q @ k.T
        s = s * jnp.exp2(jnp.minimum(da_cs[:, None] - da_cs[None, :], 0.0))
        s = jnp.where(causal_mask, s, 0.0)
        acc_o = acc_o + s @ v

        # D-skip + QK diagonal
        acc_o = acc_o + (d + qk_dot)[:, None] * v

        # State update
        da_cs_rev = da_cs_last - da_cs
        v_scaled = v * jnp.exp2(da_cs_rev)[:, None]
        new_state = ssm_state * jnp.exp2(da_cs_last) + v_scaled.T @ k

        return new_state, acc_o

    _scan_body_remat = jax.checkpoint(_scan_body)

    def _per_bh(ssm_state, v, adt, scale_ch, gamma_ch, ang, q, k, q_bias_h, k_bias_h, d):
        """Scan over chunks for a single (batch, head).
        q, k: (nchunks, chunk_size, headdim_qk) — from the GQA group
        """
        d_bc = jnp.broadcast_to(d, (nchunks,))
        final_state, out_chunks = lax.scan(
            _scan_body_remat, ssm_state,
            (q, k, v, adt, scale_ch, gamma_ch, ang, q_bias_h, k_bias_h, d_bc),
        )
        return final_state, out_chunks

    # Broadcast Q, K across GQA groups.
    # q_c: (B, nheads_qk, C, L, hqk) -> (B, nheads, C, L, hqk) via repeat
    # But we want to avoid materializing the repeat. Use vmap in_axes=None trick:
    # vmap over nheads, but Q/K are indexed by head_idx // gqa_ratio.
    # Since vmap doesn't support computed indexing, we repeat along axis 1.
    # However, jnp.repeat of (B, 1, C, L, hqk) by 32 is cheap if nheads_qk=1
    # because it's just a broadcast that XLA can handle without copying.
    if gqa_ratio > 1:
        q_c = jnp.repeat(q_c, gqa_ratio, axis=1)
        k_c = jnp.repeat(k_c, gqa_ratio, axis=1)

    # Q_bias, K_bias: (nheads, headdim_qk) -> broadcast into scan
    # Expand to (nheads, nchunks, headdim_qk) for scan input
    q_bias_expanded = jnp.broadcast_to(
        Q_bias[:, None, :], (nheads, nchunks, headdim_qk)
    )
    k_bias_expanded = jnp.broadcast_to(
        K_bias[:, None, :], (nheads, nchunks, headdim_qk)
    )

    # vmap over heads, then batch
    _vmap_heads = jax.vmap(
        _per_bh,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    _vmap_batch = jax.vmap(
        _vmap_heads,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None, None),
    )

    final_states, out_c = _vmap_batch(
        init_ssm_state,      # (B, nheads, hv, hqk)
        v_c,                 # (B, nheads, C, L, hv)
        adt_c,               # (B, nheads, C, L)
        scale_c,             # (B, nheads, C, L)
        gamma_c,             # (B, nheads, C, L)
        ang_c,               # (B, nheads, C, L, ha)
        q_c,                 # (B, nheads, C, L, hqk)
        k_c,                 # (B, nheads, C, L, hqk)
        q_bias_expanded,     # (nheads, C, hqk)
        k_bias_expanded,     # (nheads, C, hqk)
        D,                   # (nheads,)
    )

    # out_c: (B, nheads, nchunks, chunk_size, headdim_v) -> (B, seqlen, nheads, hv)
    out = jnp.transpose(out_c, (0, 2, 3, 1, 4)).reshape(batch, seqlen, nheads, headdim_v)

    if Z is not None:
        out = out * jax.nn.silu(Z.astype(jnp.float32))

    return out, final_states


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
    init_ssm_state: Optional[jnp.ndarray] = None,
    init_k_state: Optional[jnp.ndarray] = None,
    init_v_state: Optional[jnp.ndarray] = None,
    init_angle_state: Optional[jnp.ndarray] = None,
    chunk_size: int = 64,
    return_final_states: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
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
        init_*:    Initial states for recurrent inference, or None
        chunk_size: chunk size (default 64). Must be passed via functools.partial for jit.
        return_final_states: whether to return final states

    Returns:
        out: (batch, seqlen, nheads, headdim_v)
        If return_final_states, also returns:
            final_angle_state, final_ssm_state, final_k_state, final_v_state
    """
    batch, seqlen, nheads, headdim_v = V.shape
    headdim_qk = Q.shape[3]

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
        padded_seqlen = seqlen + pad_len
    else:
        padded_seqlen = seqlen

    # Cast to bf16 for compute (matching Triton kernel)
    Q = Q.astype(jnp.bfloat16)
    K = K.astype(jnp.bfloat16)
    V = V.astype(jnp.bfloat16)
    Angles = Angles.astype(jnp.bfloat16)
    if Z is not None:
        Z = Z.astype(jnp.bfloat16)

    # 1. Angle-DT cumsum (small: only headdim_angles wide)
    angles_cumsum, final_angle_state = angle_dt_cumsum(
        Angles, DT, init_state=init_angle_state, chunk_size=chunk_size,
    )

    # 2. Scale/gamma from DT, Trap (cheap, scalar per position per head)
    scale, gamma = _compute_scale_gamma(DT, Trap)

    # 3. Handle initial state trapezoidal step
    if init_ssm_state is not None and init_k_state is not None and init_v_state is not None:
        dt0 = DT[:, :, 0:1]
        trap0 = jax.nn.sigmoid(Trap[:, :, 0:1].astype(jnp.float32))
        ssm_state_init = init_ssm_state + (
            init_v_state[..., None] * init_k_state[..., None, :]
            * (dt0[..., None] * (1 - trap0[..., None]))
        )
    else:
        ssm_state_init = jnp.zeros(
            (batch, nheads, headdim_v, headdim_qk), dtype=jnp.float32
        )

    d_val = D if D is not None else jnp.zeros(nheads, dtype=jnp.float32)

    # 4. Core fused scan
    out, final_ssm_state = _mamba3_scan(
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias,
        angles_cumsum, scale, gamma, d_val, Z,
        ssm_state_init, chunk_size,
    )

    # Trim padding
    if remainder != 0:
        out = out[:, :seqlen]

    if not return_final_states:
        return out

    final_v_state = V[:, seqlen - 1]

    # Final K state: rotated+biased K at last position (pre-scale)
    scale_last = scale[:, :, seqlen - 1]  # (batch, nheads)
    # We need to recompute K_rot for the last position only
    nheads_qk = Q.shape[2]
    gqa_ratio = nheads // nheads_qk
    k_last = K[:, seqlen - 1]  # (batch, nheads_qk, headdim_qk)
    if gqa_ratio > 1:
        k_last = jnp.repeat(k_last, gqa_ratio, axis=1)
    k_last = k_last + K_bias
    ang_last = angles_cumsum[:, seqlen - 1]  # (batch, nheads, headdim_angles)
    cos_last = jnp.cos(ang_last.astype(jnp.float32))
    sin_last = jnp.sin(ang_last.astype(jnp.float32))
    # Apply rotary per head
    final_k_state = jax.vmap(jax.vmap(_apply_rotary))(k_last, cos_last, sin_last)

    return out, final_angle_state, final_ssm_state, final_k_state, final_v_state

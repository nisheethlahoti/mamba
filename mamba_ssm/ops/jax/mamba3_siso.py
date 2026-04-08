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

    if init_state is None:
        init_state = jnp.zeros((BH, dim), dtype=jnp.float32)
    else:
        init_state = init_state.reshape(BH, dim)

    final_state, out = lax.scan(_angle_dt_scan_body, init_state, vals)
    # out: (nchunks, BH, chunk_size, dim) -> (batch, nheads, nchunks, chunk_size, dim)
    out = jnp.transpose(out, (1, 0, 2, 3)).reshape(batch, nheads, nchunks, chunk_size, dim)
    # -> (batch, seqlen, nheads, dim)
    out = jnp.transpose(out, (0, 2, 3, 1, 4)).reshape(batch, seqlen, nheads, dim)
    final_state = final_state.reshape(batch, nheads, dim)
    return out, final_state



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
        [jax.nn.sigmoid(Trap[:, :, 1:].astype(jnp.float32)),
         jnp.zeros_like(trap_sig[:, :, :1])], axis=-1
    )
    gamma = dt_f32 * trap_sig
    scale = dt_shifted * (1 - trap_sig_shifted) + gamma
    return scale, gamma


# ---------------------------------------------------------------------------
# Parallel intra-chunk computation + sequential state scan
# ---------------------------------------------------------------------------

def _mamba3_scan(
    Q, K, V, ADT, Q_bias, K_bias,
    angles_cos, angles_sin,
    scale, gamma, D, Z,
    init_ssm_state,
    chunk_size,
):
    """Chunked SSM with parallel intra-chunk and sequential inter-chunk.

    The intra-chunk causal attention (Q@K^T masked, D-skip, QK diagonal) is
    independent of the recurrent state and computed in parallel across all
    chunks. Only the inter-chunk state interaction (Q @ state^T) and state
    update run through lax.scan.
    """
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    nheads = V.shape[2]
    headdim_v = V.shape[3]
    nchunks = seqlen // chunk_size
    BH = batch * nheads
    gqa_ratio = nheads // nheads_qk

    # --- Reshape helpers ---
    def _to_chunks(x_blhd):
        """(batch, seqlen, nheads, dim) -> (nchunks, BH, chunk_size, dim)"""
        x = x_blhd.reshape(batch, nchunks, chunk_size, nheads, -1)
        x = jnp.transpose(x, (1, 0, 3, 2, 4))
        return x.reshape(nchunks, BH, chunk_size, -1)

    def _to_chunks_bhl(x):
        """(batch, nheads, seqlen) -> (nchunks, BH, chunk_size)"""
        return jnp.transpose(
            x.reshape(batch, nheads, nchunks, chunk_size), (2, 0, 1, 3)
        ).reshape(nchunks, BH, chunk_size)

    # --- Preprocess all chunks in parallel ---
    # GQA expand Q, K: (batch, seqlen, nheads_qk, hqk) -> (batch, seqlen, nheads, hqk)
    if gqa_ratio > 1:
        Q = jnp.repeat(Q, gqa_ratio, axis=2)
        K = jnp.repeat(K, gqa_ratio, axis=2)

    # Bias
    Q = Q + Q_bias[None, None, :, :]
    K = K + K_bias[None, None, :, :]

    # QK dot (before rotary): (batch, seqlen, nheads)
    qk_dot = jnp.sum(Q * K, axis=-1)
    qk_dot = jnp.transpose(qk_dot, (0, 2, 1)) * gamma  # (batch, nheads, seqlen)

    # Rotary
    def _apply_rotary(x, cos, sin):
        headdim = x.shape[-1]
        n_rot = cos.shape[-1]
        pairs = x.reshape(*x.shape[:-1], headdim // 2, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]
        ro0 = x0[..., :n_rot] * cos - x1[..., :n_rot] * sin
        ro1 = x0[..., :n_rot] * sin + x1[..., :n_rot] * cos
        out0 = jnp.concatenate([ro0, x0[..., n_rot:]], axis=-1)
        out1 = jnp.concatenate([ro1, x1[..., n_rot:]], axis=-1)
        return jnp.stack([out0, out1], axis=-1).reshape(x.shape)

    Q = _apply_rotary(Q, angles_cos, angles_sin)
    K = _apply_rotary(K, angles_cos, angles_sin)

    # Scale K
    K = K * jnp.transpose(scale, (0, 2, 1))[..., None]

    # --- Chunk and compute intra-chunk output in parallel ---
    q_s = _to_chunks(Q)      # (nchunks, BH, chunk_size, hqk)
    k_s = _to_chunks(K)      # (nchunks, BH, chunk_size, hqk)
    v_s = _to_chunks(V)      # (nchunks, BH, chunk_size, hv)
    adt_s = _to_chunks_bhl(ADT)   # (nchunks, BH, chunk_size)
    qkd_s = _to_chunks_bhl(qk_dot)

    d_bh = jnp.tile(D[None, :], (batch, 1)).reshape(BH)

    # Decay per chunk
    da_s = adt_s * LOG2E                       # (nchunks, BH, chunk_size)
    da_cs_s = jnp.cumsum(da_s, axis=-1)        # (nchunks, BH, chunk_size)

    # Intra-chunk causal attention (parallel over all chunks)
    causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=-1)
    s = jnp.matmul(q_s, jnp.transpose(k_s, (0, 1, 3, 2)))  # (nchunks, BH, L, L)
    s = s * jnp.exp2(jnp.minimum(da_cs_s[:, :, :, None] - da_cs_s[:, :, None, :], 0.0))
    s = jnp.where(causal_mask[None, None, :, :], s, 0.0)
    intra_out = jnp.matmul(s, v_s)  # (nchunks, BH, L, hv)

    # D-skip + QK diagonal
    intra_out = intra_out + (d_bh[None, :, None] + qkd_s)[:, :, :, None] * v_s

    # --- Sequential scan: only inter-chunk state interaction ---
    state_init = init_ssm_state.reshape(BH, headdim_v, headdim_qk)

    def _scan_body(ssm_state, inputs):
        q, k, v, da_cs = inputs
        da_last = da_cs[:, -1]

        # Inter-chunk output: Q @ State^T * exp2(da_cs)
        inter_out = jnp.matmul(q, jnp.transpose(ssm_state, (0, 2, 1)))
        inter_out = inter_out * jnp.exp2(da_cs)[:, :, None]

        # State update
        v_scaled = v * jnp.exp2(da_last[:, None] - da_cs)[:, :, None]
        new_state = (ssm_state * jnp.exp2(da_last)[:, None, None]
                     + jnp.matmul(jnp.transpose(v_scaled, (0, 2, 1)), k))

        return new_state, inter_out

    final_state, inter_out_s = lax.scan(
        jax.checkpoint(_scan_body), state_init, (q_s, k_s, v_s, da_cs_s))

    # Combine intra + inter
    out_s = intra_out + inter_out_s

    # Reshape: (nchunks, BH, chunk_size, hv) -> (batch, seqlen, nheads, hv)
    out = out_s.reshape(nchunks, batch, nheads, chunk_size, headdim_v)
    out = jnp.transpose(out, (1, 0, 3, 2, 4)).reshape(batch, seqlen, nheads, headdim_v)

    if Z is not None:
        out = out * jax.nn.silu(Z.astype(jnp.float32))

    final_state = final_state.reshape(batch, nheads, headdim_v, headdim_qk)
    return out, final_state


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

    # 1. Angle-DT cumsum
    angles_cumsum, final_angle_state = angle_dt_cumsum(
        Angles, DT, init_state=init_angle_state, chunk_size=chunk_size,
    )

    # 2. Precompute cos/sin of angles (small, avoids trig in scan body)
    angles_cos = jnp.cos(angles_cumsum.astype(jnp.float32))
    angles_sin = jnp.sin(angles_cumsum.astype(jnp.float32))

    # 3. Scale/gamma
    scale, gamma = _compute_scale_gamma(DT, Trap)

    # 4. Initial state
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

    # 5. Core scan
    out, final_ssm_state = _mamba3_scan(
        Q, K, V, ADT, Q_bias, K_bias,
        angles_cos, angles_sin,
        scale, gamma, d_val, Z,
        ssm_state_init, chunk_size,
    )

    # Trim padding
    if remainder != 0:
        out = out[:, :seqlen]

    if not return_final_states:
        return out

    final_v_state = V[:, seqlen - 1]

    # Final K state: apply bias + rotary at last position
    nheads_qk = Q.shape[2]
    gqa_ratio = nheads // nheads_qk
    k_last = K[:, seqlen - 1]  # (batch, nheads_qk, hqk)
    if gqa_ratio > 1:
        k_last = jnp.repeat(k_last, gqa_ratio, axis=1)
    k_last = k_last + K_bias
    cos_last = angles_cos[:, seqlen - 1]  # (batch, nheads, ha)
    sin_last = angles_sin[:, seqlen - 1]
    n_rot = cos_last.shape[-1]
    pairs = k_last.reshape(batch, nheads, headdim_qk // 2, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    ro0 = x0[..., :n_rot] * cos_last - x1[..., :n_rot] * sin_last
    ro1 = x0[..., :n_rot] * sin_last + x1[..., :n_rot] * cos_last
    out0 = jnp.concatenate([ro0, x0[..., n_rot:]], axis=-1)
    out1 = jnp.concatenate([ro1, x1[..., n_rot:]], axis=-1)
    final_k_state = jnp.stack([out0, out1], axis=-1).reshape(batch, nheads, headdim_qk)

    return out, final_angle_state, final_ssm_state, final_k_state, final_v_state

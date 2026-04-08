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
    """Scan body for a single (batch, head) slice.

    state: (dim,)  — cumulative angle
    chunk: (chunk_size, dim) — tanh(angle)*pi*dt for this chunk
    """
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
        chunk_size: chunk size

    Returns:
        out:         (batch, seqlen, nheads, dim)
        final_state: (batch, nheads, dim)
    """
    batch, seqlen, nheads, dim = angles.shape
    nchunks = seqlen // chunk_size

    # (batch, nheads, nchunks, chunk_size, dim)
    vals = jnp.tanh(angles.astype(jnp.float32)) * math.pi
    # dt: (batch, nheads, seqlen) -> (batch, nheads, nchunks, chunk_size)
    dt_r = dt.reshape(batch, nheads, nchunks, chunk_size)
    # angles: (batch, seqlen, nheads, dim) -> (batch, nheads, nchunks, chunk_size, dim)
    vals = jnp.transpose(vals.reshape(batch, nchunks, chunk_size, nheads, dim), (0, 3, 1, 2, 4))
    vals = vals * dt_r[..., None]

    if init_state is None:
        init_state = jnp.zeros((batch, nheads, dim), dtype=jnp.float32)

    # vmap over (batch, nheads), scan over nchunks
    def _per_bh(state, chunks):
        # state: (dim,), chunks: (nchunks, chunk_size, dim)
        final_state, out = lax.scan(_angle_dt_scan_body, state, chunks)
        return final_state, out

    _batched = jax.vmap(jax.vmap(_per_bh, in_axes=(0, 0)), in_axes=(0, 0))
    final_state, out = _batched(init_state, vals)

    # out: (batch, nheads, nchunks, chunk_size, dim) -> (batch, seqlen, nheads, dim)
    out = jnp.transpose(out, (0, 2, 3, 1, 4)).reshape(batch, seqlen, nheads, dim)
    return out, final_state


# ---------------------------------------------------------------------------
# Preprocessing: bias, rotary, scale
# ---------------------------------------------------------------------------

def _apply_rotary(x, cos, sin, headdim_angles):
    """Apply rotary embedding to x. Only first headdim_angles pairs are rotated."""
    headdim = x.shape[-1]
    n_rot = headdim_angles  # number of pairs to rotate
    # Split into pairs: reshape last dim to (..., headdim//2, 2)
    x_pairs = x.reshape(*x.shape[:-1], headdim // 2, 2)
    x0 = x_pairs[..., 0]  # (..., headdim//2)
    x1 = x_pairs[..., 1]

    ro0 = x0[..., :n_rot] * cos - x1[..., :n_rot] * sin
    ro1 = x0[..., :n_rot] * sin + x1[..., :n_rot] * cos

    out0 = jnp.concatenate([ro0, x0[..., n_rot:]], axis=-1)
    out1 = jnp.concatenate([ro1, x1[..., n_rot:]], axis=-1)
    return jnp.stack([out0, out1], axis=-1).reshape(*x.shape[:-1], headdim)


def preprocess_qk(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    Q_bias: jnp.ndarray,
    K_bias: jnp.ndarray,
    DT: jnp.ndarray,
    Trap: jnp.ndarray,
    angles_cumsum: jnp.ndarray,
    nheads: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Preprocess Q and K: bias, rotary, scale. All elementwise.

    Args:
        Q, K:            (batch, seqlen, nheads_qk, headdim_qk)
        Q_bias, K_bias:  (nheads, headdim_qk)
        DT:              (batch, nheads, seqlen)  — already softplus'd
        Trap:            (batch, nheads, seqlen)  — raw, pre-sigmoid
        angles_cumsum:   (batch, seqlen, nheads, headdim_angles)
        nheads:          number of output heads

    Returns:
        Q_rot:    (batch, seqlen, nheads, headdim_qk)
        K_scaled: (batch, seqlen, nheads, headdim_qk)
        qk_dot:   (batch, nheads, seqlen)
    """
    nheads_qk = Q.shape[2]
    ratio = nheads // nheads_qk

    # GQA expansion
    if ratio > 1:
        Q = jnp.repeat(Q, ratio, axis=2)
        K = jnp.repeat(K, ratio, axis=2)

    # Add biases
    Q = Q + Q_bias[None, None, :, :]
    K = K + K_bias[None, None, :, :]

    # Compute scale and gamma from DT, Trap
    # DT, Trap: (batch, nheads, seqlen)
    trap_sig = jax.nn.sigmoid(Trap.astype(jnp.float32))
    dt_f32 = DT.astype(jnp.float32)

    # Shifted versions: dt[t+1], trap[t+1]; 0 at the boundary
    dt_shifted = jnp.concatenate([dt_f32[:, :, 1:], jnp.zeros_like(dt_f32[:, :, :1])], axis=-1)
    trap_sig_shifted = jnp.concatenate([
        jax.nn.sigmoid(Trap[:, :, 1:].astype(jnp.float32)),
        jnp.zeros_like(trap_sig[:, :, :1])
    ], axis=-1)

    gamma = dt_f32 * trap_sig                        # (batch, nheads, seqlen)
    scale = dt_shifted * (1 - trap_sig_shifted) + gamma  # (batch, nheads, seqlen)

    # QK dot product (before rotary, matching Triton kernel)
    # sum over headdim, then multiply by gamma
    qk_dot = jnp.sum(Q * K, axis=-1)  # (batch, seqlen, nheads)
    qk_dot = jnp.transpose(qk_dot, (0, 2, 1)) * gamma  # (batch, nheads, seqlen)

    # Rotary embeddings
    headdim_angles = angles_cumsum.shape[-1]
    cos_a = jnp.cos(angles_cumsum.astype(jnp.float32))
    sin_a = jnp.sin(angles_cumsum.astype(jnp.float32))
    Q_rot = _apply_rotary(Q, cos_a, sin_a, headdim_angles)
    K_rot = _apply_rotary(K, cos_a, sin_a, headdim_angles)

    # Scale K: (batch, nheads, seqlen) -> (batch, seqlen, nheads, 1)
    scale_expanded = jnp.transpose(scale, (0, 2, 1))[..., None]
    K_scaled = K_rot * scale_expanded

    return Q_rot, K_scaled, qk_dot


# ---------------------------------------------------------------------------
# Core chunked SSM scan
# ---------------------------------------------------------------------------

def mamba3_siso_chunk_scan(
    Q_rot: jnp.ndarray,
    K_scaled: jnp.ndarray,
    V: jnp.ndarray,
    ADT: jnp.ndarray,
    qk_dot: jnp.ndarray,
    D: Optional[jnp.ndarray],
    Z: Optional[jnp.ndarray],
    init_ssm_state: Optional[jnp.ndarray] = None,
    chunk_size: int = 64,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Chunked SSM via jax.lax.scan. The core computation.

    Args:
        Q_rot:          (batch, seqlen, nheads, headdim_qk)
        K_scaled:       (batch, seqlen, nheads, headdim_qk)
        V:              (batch, seqlen, nheads, headdim_v)
        ADT:            (batch, nheads, seqlen) — A*dt decay
        qk_dot:         (batch, nheads, seqlen)
        D:              (nheads,) or None
        Z:              (batch, seqlen, nheads, headdim_v) or None
        init_ssm_state: (batch, nheads, headdim_v, headdim_qk) or None
        chunk_size:     chunk size

    Returns:
        out:             (batch, seqlen, nheads, headdim_v)
        final_ssm_state: (batch, nheads, headdim_v, headdim_qk)
    """
    batch, seqlen, nheads, headdim_qk = Q_rot.shape
    headdim_v = V.shape[-1]
    nchunks = seqlen // chunk_size

    # Reshape to (batch, nheads, nchunks, chunk_size, hdim)
    def _to_chunks(x):
        # x: (batch, seqlen, nheads, hdim) -> (batch, nheads, nchunks, chunk_size, hdim)
        return jnp.transpose(
            x.reshape(batch, nchunks, chunk_size, nheads, -1),
            (0, 3, 1, 2, 4),
        )

    q_c = _to_chunks(Q_rot)
    k_c = _to_chunks(K_scaled)
    v_c = _to_chunks(V)
    # ADT, qk_dot: (batch, nheads, seqlen) -> (batch, nheads, nchunks, chunk_size)
    adt_c = ADT.reshape(batch, nheads, nchunks, chunk_size)
    qkd_c = qk_dot.reshape(batch, nheads, nchunks, chunk_size)

    if init_ssm_state is None:
        init_ssm_state = jnp.zeros((batch, nheads, headdim_v, headdim_qk), dtype=jnp.float32)

    d_val = D if D is not None else jnp.zeros(nheads, dtype=jnp.float32)

    # Causal mask: strictly lower triangular (matches Triton's i > j condition)
    causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=-1)

    def _scan_body(ssm_state, inputs):
        """Process one chunk for a single (batch, head).

        ssm_state: (headdim_v, headdim_qk)
        inputs: q, k, v (chunk_size, hdim), adt/qkd (chunk_size,), d_val scalar
        """
        q, k, v, adt, qkd, d = inputs

        da = adt * LOG2E
        da_cs = jnp.cumsum(da)
        da_cs_last = jnp.sum(da)

        # Inter-chunk contribution: Q @ State^T * exp2(da_cs)
        acc_o = (q @ ssm_state.T) * jnp.exp2(da_cs)[:, None]

        # Intra-chunk: causal(Q @ K^T * exp2(decay)) @ V
        s = q @ k.T
        s = s * jnp.exp2(jnp.minimum(da_cs[:, None] - da_cs[None, :], 0.0))
        s = jnp.where(causal_mask, s, 0.0)
        acc_o = acc_o + s @ v

        # D-skip + QK diagonal
        acc_o = acc_o + (d + qkd)[:, None] * v

        # State update
        da_cs_rev = da_cs_last - da_cs
        v_scaled = v * jnp.exp2(da_cs_rev)[:, None]
        new_state = ssm_state * jnp.exp2(da_cs_last) + v_scaled.T @ k

        return new_state, acc_o

    _scan_body_remat = jax.checkpoint(_scan_body)

    def _per_bh(ssm_state, q, k, v, adt, qkd, d):
        """Scan over chunks for a single (batch, head)."""
        # q,k,v: (nchunks, chunk_size, hdim), adt/qkd: (nchunks, chunk_size)
        d_broadcast = jnp.broadcast_to(d, (nchunks,))
        final_state, out_chunks = lax.scan(
            _scan_body_remat, ssm_state, (q, k, v, adt, qkd, d_broadcast)
        )
        return final_state, out_chunks

    # vmap over nheads then batch
    _vmap_heads = jax.vmap(_per_bh, in_axes=(0, 0, 0, 0, 0, 0, 0))
    _vmap_batch = jax.vmap(_vmap_heads, in_axes=(0, 0, 0, 0, 0, 0, None))

    final_states, out_c = _vmap_batch(
        init_ssm_state, q_c, k_c, v_c, adt_c, qkd_c, d_val
    )
    # out_c: (batch, nheads, nchunks, chunk_size, headdim_v)
    # -> (batch, seqlen, nheads, headdim_v)
    out = jnp.transpose(out_c, (0, 2, 3, 1, 4)).reshape(batch, seqlen, nheads, headdim_v)

    # Z-gating (applied outside scan — Z doesn't affect state update)
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
        ADT:       (batch, nheads, seqlen) — A*dt decay factor
        DT:        (batch, nheads, seqlen) — time delta (post-softplus)
        Trap:      (batch, nheads, seqlen) — trapezoidal factor (raw, pre-sigmoid)
        Q_bias:    (nheads, headdim_qk)
        K_bias:    (nheads, headdim_qk)
        Angles:    (batch, seqlen, nheads, headdim_angles)
        D:         (nheads,) or None — skip connection
        Z:         (batch, seqlen, nheads, headdim_v) or None — gating
        init_*:    Initial states for recurrent inference, or None
        chunk_size: chunk size (default 64)
        return_final_states: whether to return final states

    Returns:
        out: (batch, seqlen, nheads, headdim_v)
        If return_final_states, also returns:
            final_angle_state, final_ssm_state, final_k_state, final_v_state
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

    # 2. Preprocess Q, K
    Q_rot, K_scaled, qk_dot = preprocess_qk(
        Q, K, Q_bias, K_bias, DT, Trap, angles_cumsum, nheads,
    )

    # 3. Handle initial state trapezoidal step
    ssm_state_init = init_ssm_state
    if init_ssm_state is not None and init_k_state is not None and init_v_state is not None:
        dt0 = DT[:, :, 0:1]  # (batch, nheads, 1)
        trap0 = jax.nn.sigmoid(Trap[:, :, 0:1].astype(jnp.float32))
        # state += v_state[:, None] * k_state[None, :] * dt * (1 - trap)
        # v_state: (batch, nheads, headdim_v), k_state: (batch, nheads, headdim_qk)
        ssm_state_init = init_ssm_state + (
            init_v_state[..., None] * init_k_state[..., None, :]
            * (dt0[..., None] * (1 - trap0[..., None]))
        )

    # 4. Core SSM scan
    out, final_ssm_state = mamba3_siso_chunk_scan(
        Q_rot, K_scaled, V, ADT, qk_dot, D, Z,
        init_ssm_state=ssm_state_init,
        chunk_size=chunk_size,
    )

    # Trim padding
    if remainder != 0:
        out = out[:, :seqlen]

    if not return_final_states:
        return out

    # Final K state: last K_rot (pre-scale) at the last position
    # Final V state: last V at the last position
    final_v_state = V[:, seqlen - 1]  # (batch, nheads, headdim_v)

    # For final_k_state, we need the rotated+biased K *before* scaling
    # Recompute just the last position's K_rot from K_scaled / scale
    # This is simpler than storing it separately
    scale_last = jnp.transpose(
        _compute_scale(DT, Trap)[:, :, seqlen - 1:seqlen], (0, 2, 1)
    )[..., None]  # (batch, 1, nheads, 1)
    final_k_state = K_scaled[:, seqlen - 1:seqlen] / (scale_last + 1e-8)
    final_k_state = final_k_state.squeeze(1)  # (batch, nheads, headdim_qk)

    return out, final_angle_state, final_ssm_state, final_k_state, final_v_state


def _compute_scale(DT, Trap):
    """Recompute scale factor. DT, Trap: (batch, nheads, seqlen)."""
    dt_f32 = DT.astype(jnp.float32)
    trap_sig = jax.nn.sigmoid(Trap.astype(jnp.float32))
    dt_shifted = jnp.concatenate([dt_f32[:, :, 1:], jnp.zeros_like(dt_f32[:, :, :1])], axis=-1)
    trap_sig_shifted = jnp.concatenate([
        jax.nn.sigmoid(Trap[:, :, 1:].astype(jnp.float32)),
        jnp.zeros_like(trap_sig[:, :, :1]),
    ], axis=-1)
    gamma = dt_f32 * trap_sig
    return dt_shifted * (1 - trap_sig_shifted) + gamma

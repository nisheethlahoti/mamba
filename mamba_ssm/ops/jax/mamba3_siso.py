"""Mamba-3 SISO in JAX with Pallas Triton-fused SSM core.

Preprocessing (bias, rotary, GQA, scale) is vectorized across all positions.
Each per-chunk SSM body (cumsum, matmuls, exp2, causal mask) is fused into a
single Pallas Triton kernel, called from within a vmap'd lax.scan over chunks.
Backward pass uses the pure-JAX scan via custom_vjp.

Falls back to pure JAX on CPU or when Pallas is unavailable.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pl_triton
    _HAS_PALLAS = True
except ImportError:
    _HAS_PALLAS = False

LOG2E = math.log2(math.e)
TWO_PI = 2 * math.pi


# ---------------------------------------------------------------------------
# angle_dt cumsum (small sequential scan over headdim_angles)
# ---------------------------------------------------------------------------

def _angle_dt_scan_body(state, chunk):
    chunk_cumsum = jnp.cumsum(chunk, axis=0)
    out = chunk_cumsum + state[None, :]
    out = out - TWO_PI * jnp.floor(out / TWO_PI)
    new_state = state + jnp.sum(chunk, axis=0)
    new_state = new_state - TWO_PI * jnp.floor(new_state / TWO_PI)
    return new_state, out


def angle_dt_cumsum(angles, dt, init_state=None, chunk_size=64):
    """cumsum(tanh(angles) * pi * dt) mod 2pi, chunked."""
    batch, seqlen, nheads, dim = angles.shape
    nchunks = seqlen // chunk_size
    vals = jnp.tanh(angles.astype(jnp.float32)) * math.pi
    dt_r = dt.reshape(batch, nheads, nchunks, chunk_size)
    vals = jnp.transpose(
        vals.reshape(batch, nchunks, chunk_size, nheads, dim), (0, 3, 1, 2, 4))
    vals = vals * dt_r[..., None]
    if init_state is None:
        init_state = jnp.zeros((batch, nheads, dim), dtype=jnp.float32)
    def _per_bh(state, chunks):
        return lax.scan(_angle_dt_scan_body, state, chunks)
    final_state, out = jax.vmap(jax.vmap(_per_bh))(init_state, vals)
    out = jnp.transpose(out, (0, 2, 3, 1, 4)).reshape(batch, seqlen, nheads, dim)
    return out, final_state


# ---------------------------------------------------------------------------
# Preprocessing: bias, rotary, scale, qk_dot (vectorized, no scan)
# ---------------------------------------------------------------------------

def _apply_rotary(x, cos, sin):
    """x: (L, headdim), cos/sin: (L, n_rot). Used for final_k_state."""
    headdim = x.shape[-1]
    n_rot = cos.shape[-1]
    pairs = x.reshape(-1, headdim // 2, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    ro0 = x0[..., :n_rot] * cos - x1[..., :n_rot] * sin
    ro1 = x0[..., :n_rot] * sin + x1[..., :n_rot] * cos
    out0 = jnp.concatenate([ro0, x0[..., n_rot:]], axis=-1)
    out1 = jnp.concatenate([ro1, x1[..., n_rot:]], axis=-1)
    return jnp.stack([out0, out1], axis=-1).reshape(-1, headdim)


def _apply_rotary_batch(x, cos, sin):
    """Rotary for arbitrary batch dims. x: (..., headdim), cos/sin: (..., n_rot)."""
    headdim = x.shape[-1]
    n_rot = cos.shape[-1]
    pairs = x.reshape(*x.shape[:-1], headdim // 2, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    ro0 = x0[..., :n_rot] * cos - x1[..., :n_rot] * sin
    ro1 = x0[..., :n_rot] * sin + x1[..., :n_rot] * cos
    out0 = jnp.concatenate([ro0, x0[..., n_rot:]], axis=-1)
    out1 = jnp.concatenate([ro1, x1[..., n_rot:]], axis=-1)
    return jnp.stack([out0, out1], axis=-1).reshape(x.shape)


def _preprocess(Q, K, Q_bias, K_bias, DT, Trap, angles_cumsum, nheads):
    """Vectorized preprocessing: bias, qk_dot, rotary, scale."""
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    gqa_ratio = nheads // nheads_qk

    if gqa_ratio > 1:
        Q = jnp.repeat(Q, gqa_ratio, axis=2)
        K = jnp.repeat(K, gqa_ratio, axis=2)

    Q = Q + Q_bias[None, None, :, :]
    K = K + K_bias[None, None, :, :]

    # Scale / gamma from DT, Trap
    dt_f32 = DT.astype(jnp.float32)
    trap_sig = jax.nn.sigmoid(Trap.astype(jnp.float32))
    dt_shifted = jnp.concatenate(
        [dt_f32[:, :, 1:], jnp.zeros_like(dt_f32[:, :, :1])], axis=-1)
    trap_sig_shifted = jnp.concatenate(
        [jax.nn.sigmoid(Trap[:, :, 1:].astype(jnp.float32)),
         jnp.zeros_like(trap_sig[:, :, :1])], axis=-1)
    gamma = dt_f32 * trap_sig
    scale = dt_shifted * (1 - trap_sig_shifted) + gamma

    # QK dot (before rotary)
    gamma_blh = jnp.transpose(gamma, (0, 2, 1))
    qk_dot = jnp.sum(Q * K, axis=-1) * gamma_blh

    # Rotary
    cos_a = jnp.cos(angles_cumsum.astype(jnp.float32))
    sin_a = jnp.sin(angles_cumsum.astype(jnp.float32))
    Q = _apply_rotary_batch(Q, cos_a, sin_a)
    K = _apply_rotary_batch(K, cos_a, sin_a)

    # Scale K
    scale_blh = jnp.transpose(scale, (0, 2, 1))
    K = K * scale_blh[..., None]

    return Q, K, qk_dot


# ---------------------------------------------------------------------------
# Pure JAX scan (backward + CPU fallback)
# ---------------------------------------------------------------------------

def _scan_body_jax(causal_mask, ssm_state, inputs):
    """One chunk for a single (batch, head). Pure JAX. SSM core only."""
    q, k, v, adt, qk_dot, d = inputs

    da = adt * LOG2E
    da_cs = jnp.cumsum(da)
    da_cs_last = jnp.sum(da)

    acc_o = (q @ ssm_state.T) * jnp.exp2(da_cs)[:, None]
    s = q @ k.T
    s = s * jnp.exp2(jnp.minimum(da_cs[:, None] - da_cs[None, :], 0.0))
    s = jnp.where(causal_mask, s, 0.0)
    acc_o = acc_o + s @ v
    acc_o = acc_o + (d + qk_dot)[:, None] * v

    v_scaled = v * jnp.exp2(da_cs_last - da_cs)[:, None]
    new_state = ssm_state * jnp.exp2(da_cs_last) + v_scaled.T @ k
    return new_state, acc_o


def _ssm_scan_jax(init_state, scan_inputs, chunk_size):
    """Pure JAX chunked SSM via vmap'd lax.scan."""
    causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=-1)

    def _body(state, inputs):
        return _scan_body_jax(causal_mask, state, inputs)

    def _per_bh(state, inputs):
        return lax.scan(jax.checkpoint(_body), state, inputs)

    _vmap_heads = jax.vmap(_per_bh)
    _vmap_batch = jax.vmap(_vmap_heads)
    return _vmap_batch(init_state, scan_inputs)


# ---------------------------------------------------------------------------
# Pallas SSM kernel (GPU forward — SSM core only)
# ---------------------------------------------------------------------------

def _make_pallas_chunk_body(chunk_sz, headdim_qk, headdim_v):
    """Build a Pallas Triton kernel for one SSM chunk, for use inside lax.scan."""

    def kernel(q_ref, k_ref, v_ref, adt_ref, qkd_ref,
               state_ref, out_ref, fstate_ref):
        state = state_ref[:, :]
        causal_mask = jnp.tril(jnp.ones((chunk_sz, chunk_sz), dtype=jnp.float32), k=-1)

        q = q_ref[:, :].astype(jnp.float32)
        k = k_ref[:, :].astype(jnp.float32)
        v = v_ref[:, :].astype(jnp.float32)
        adt = adt_ref[:]
        qkd = qkd_ref[:]

        da = adt * LOG2E
        da_cs = jnp.cumsum(da)
        da_cs_last = jnp.sum(da)

        acc = jnp.dot(q, state.T) * jnp.exp2(da_cs)[:, None]
        s = jnp.dot(q, k.T)
        s = s * jnp.exp2(jnp.minimum(da_cs[:, None] - da_cs[None, :], 0.0))
        s = s * causal_mask
        acc = acc + jnp.dot(s, v)
        acc = acc + qkd[:, None] * v

        out_ref[:, :] = acc

        v_scaled = v * jnp.exp2(da_cs_last - da_cs)[:, None]
        new_state = state * jnp.exp2(da_cs_last) + jnp.dot(v_scaled.T, k)
        fstate_ref[:, :] = new_state

    def scan_body(state, inputs):
        q, k, v, adt, qkd = inputs
        out, new_state = pl.pallas_call(
            kernel,
            out_shape=[
                jax.ShapeDtypeStruct((chunk_sz, headdim_v), jnp.float32),
                jax.ShapeDtypeStruct((headdim_v, headdim_qk), jnp.float32),
            ],
            compiler_params=pl_triton.CompilerParams(),
        )(q, k, v, adt, qkd, state)
        return new_state, out

    return scan_body


def _ssm_scan_pallas(init_state, scan_inputs, chunk_size):
    """Pallas-fused per-chunk body inside vmap'd lax.scan over chunks."""
    q_c, k_c, v_c, adt_c, qkdot_c, d_c = scan_inputs
    chunk_sz = q_c.shape[3]
    headdim_qk = q_c.shape[4]
    headdim_v = v_c.shape[4]

    # Pre-add D to qk_dot
    qkdot_c = qkdot_c + d_c[:, :, :, None]

    scan_body = _make_pallas_chunk_body(chunk_sz, headdim_qk, headdim_v)

    def _per_bh(state, inputs):
        return lax.scan(scan_body, state, inputs)

    _vmap_heads = jax.vmap(_per_bh)
    _vmap_batch = jax.vmap(_vmap_heads)

    final_states, out_c = _vmap_batch(init_state, (q_c, k_c, v_c, adt_c, qkdot_c))
    return final_states, out_c


# ---------------------------------------------------------------------------
# Dispatch: Pallas on GPU (forward), JAX everywhere (backward + CPU)
# ---------------------------------------------------------------------------

def _ssm_scan_dispatch(init_state, scan_inputs, chunk_size):
    """Use Pallas for forward on GPU, JAX for backward via custom_vjp."""
    use_pallas = (_HAS_PALLAS
                  and len(jax.devices()) > 0
                  and jax.devices()[0].platform == 'gpu')

    if not use_pallas:
        return _ssm_scan_jax(init_state, scan_inputs, chunk_size)

    @jax.custom_vjp
    def _fwd(init_state, *flat_inputs):
        return _ssm_scan_pallas(init_state, flat_inputs, chunk_size)

    def _fwd_fwd(init_state, *flat_inputs):
        result = _fwd(init_state, *flat_inputs)
        return result, (init_state, flat_inputs)

    def _fwd_bwd(residuals, g):
        init_state, flat_inputs = residuals
        def jax_fn(init_state, *flat_inputs):
            return _ssm_scan_jax(init_state, flat_inputs, chunk_size)
        _, vjp_fn = jax.vjp(jax_fn, init_state, *flat_inputs)
        return vjp_fn(g)

    _fwd.defvjp(_fwd_fwd, _fwd_bwd)
    return _fwd(init_state, *scan_inputs)


# ---------------------------------------------------------------------------
# Main scan orchestration
# ---------------------------------------------------------------------------

def _mamba3_scan(Q_rot, K_scaled, V, ADT, qk_dot, D, Z,
                 init_ssm_state, chunk_size):
    """Chunk preprocessed inputs and run the SSM scan."""
    batch, seqlen, nheads, headdim_qk = Q_rot.shape
    headdim_v = V.shape[3]
    nchunks = seqlen // chunk_size

    def _chunk_bhl(x):
        return x.reshape(batch, nheads, nchunks, chunk_size)

    def _chunk_blhd(x):
        return jnp.transpose(
            x.reshape(batch, nchunks, chunk_size, x.shape[2], x.shape[3]),
            (0, 3, 1, 2, 4))

    q_c = _chunk_blhd(Q_rot)
    k_c = _chunk_blhd(K_scaled)
    v_c = _chunk_blhd(V)
    adt_c = _chunk_bhl(ADT)
    qkdot_c = jnp.transpose(qk_dot, (0, 2, 1)).reshape(
        batch, nheads, nchunks, chunk_size)
    d_expanded = jnp.broadcast_to(
        D[None, :, None], (batch, nheads, nchunks))

    scan_inputs = (q_c, k_c, v_c, adt_c, qkdot_c, d_expanded)

    final_states, out_c = _ssm_scan_dispatch(
        init_ssm_state, scan_inputs, chunk_size)

    out = jnp.transpose(out_c, (0, 2, 3, 1, 4)).reshape(
        batch, seqlen, nheads, headdim_v)

    if Z is not None:
        out = out * jax.nn.silu(Z.astype(jnp.float32))

    return out, final_states


# ---------------------------------------------------------------------------
# Entry point
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
    """Mamba-3 SISO forward pass.

    Args:
        Q:         (batch, seqlen, nheads_qk, headdim_qk)
        K:         (batch, seqlen, nheads_qk, headdim_qk)
        V:         (batch, seqlen, nheads, headdim_v)
        ADT:       (batch, nheads, seqlen)
        DT:        (batch, nheads, seqlen)
        Trap:      (batch, nheads, seqlen)
        Q_bias:    (nheads, headdim_qk)
        K_bias:    (nheads, headdim_qk)
        Angles:    (batch, seqlen, nheads, headdim_angles)
        D:         (nheads,) or None
        Z:         (batch, seqlen, nheads, headdim_v) or None
        init_*:    Initial states for recurrent inference, or None
        chunk_size: Must be passed via functools.partial for jit.
        return_final_states: whether to return final states
    """
    batch, seqlen, nheads, headdim_v = V.shape
    headdim_qk = Q.shape[3]

    remainder = seqlen % chunk_size
    if remainder != 0:
        pad_len = chunk_size - remainder
        def _pad(x, ax=1):
            w = [(0, 0)] * x.ndim
            w[ax] = (0, pad_len)
            return jnp.pad(x, w)
        Q, K, V, Angles = _pad(Q), _pad(K), _pad(V), _pad(Angles)
        ADT, DT, Trap = _pad(ADT, 2), _pad(DT, 2), _pad(Trap, 2)
        if Z is not None:
            Z = _pad(Z)

    Q = Q.astype(jnp.bfloat16)
    K = K.astype(jnp.bfloat16)
    V = V.astype(jnp.bfloat16)
    Angles = Angles.astype(jnp.bfloat16)
    if Z is not None:
        Z = Z.astype(jnp.bfloat16)

    angles_cumsum, final_angle_state = angle_dt_cumsum(
        Angles, DT, init_state=init_angle_state, chunk_size=chunk_size)

    if init_ssm_state is not None and init_k_state is not None and init_v_state is not None:
        dt0 = DT[:, :, 0:1]
        trap0 = jax.nn.sigmoid(Trap[:, :, 0:1].astype(jnp.float32))
        ssm_init = init_ssm_state + (
            init_v_state[..., None] * init_k_state[..., None, :]
            * (dt0[..., None] * (1 - trap0[..., None])))
    else:
        ssm_init = jnp.zeros((batch, nheads, headdim_v, headdim_qk), dtype=jnp.float32)

    d_val = D if D is not None else jnp.zeros(nheads, dtype=jnp.float32)

    # Vectorized preprocessing (no scan needed — all elementwise/reductions)
    Q_rot, K_scaled, qk_dot = _preprocess(
        Q, K, Q_bias, K_bias, DT, Trap, angles_cumsum, nheads)

    out, final_ssm_state = _mamba3_scan(
        Q_rot, K_scaled, V, ADT, qk_dot, d_val, Z, ssm_init, chunk_size)

    if remainder != 0:
        out = out[:, :seqlen]

    if not return_final_states:
        return out

    final_v_state = V[:, seqlen - 1]
    nheads_qk = Q.shape[2]
    gqa_ratio = nheads // nheads_qk
    k_last = K[:, seqlen - 1]
    if gqa_ratio > 1:
        k_last = jnp.repeat(k_last, gqa_ratio, axis=1)
    k_last = k_last + K_bias
    ang_last = angles_cumsum[:, seqlen - 1]
    cos_last = jnp.cos(ang_last.astype(jnp.float32))
    sin_last = jnp.sin(ang_last.astype(jnp.float32))
    final_k_state = jax.vmap(jax.vmap(_apply_rotary))(
        k_last, cos_last, sin_last).squeeze(2)

    return out, final_angle_state, final_ssm_state, final_k_state, final_v_state

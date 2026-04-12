"""Mamba-3 SISO in pure JAX"""

import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from jax import Array
from jax._src.lax.control_flow.loops import _interleave


def trace(x: Array, coeff: Array) -> Array:
    if not x.shape[0]:
        return jnp.zeros((1,) + x.shape[1:], x.dtype)
    x0, x1 = x[::2], x[1::2]
    c0, c1 = coeff[::2], coeff[1::2]
    arr = trace(x0[: x1.shape[0]] * c1 + x1, c0[: c1.shape[0]] * c1)
    return _interleave(arr, arr[: x0.shape[0]] * c0 + x0, axis=0)


def angle_dt_cumsum(angles: Array, dt: Array) -> Array:
    """Compute cumsum(tanh(angles) * pi * dt) mod 2pi.

    Args:
        angles:     (batch, seqlen, nheads, dim)
        dt:         (batch, nheads, seqlen)
    Returns:
        out:         (batch, seqlen, nheads, dim)
    """
    vals = jnp.tanh(angles.astype(jnp.float32)) * jnp.pi
    vals *= rearrange(dt, "b h s -> b s h 1")
    out = vals.cumsum(axis=1) % (2 * jnp.pi)
    return out.astype(angles.dtype)


def _apply_rotary_batched(x, cos, sin):
    """Apply rotary embedding, batched.

    x:   (BH, chunk_size, headdim_qk)
    cos: (BH, chunk_size, n_rot)
    sin: (BH, chunk_size, n_rot)
    """
    n_rot = cos.shape[-1]
    x = rearrange(x, "bh l (p two) -> bh l p two", two=2)
    x0, x1 = jnp.unstack(x, axis=-1)

    ro0 = x0[..., :n_rot] * cos - x1[..., :n_rot] * sin
    ro1 = x0[..., :n_rot] * sin + x1[..., :n_rot] * cos

    out0 = jnp.concatenate([ro0, x0[..., n_rot:]], axis=-1)
    out1 = jnp.concatenate([ro1, x1[..., n_rot:]], axis=-1)
    out = jnp.stack([out0, out1], axis=-1)
    return rearrange(out, "bh l p two -> bh l (p two)")


def _compute_scale_gamma(DT, Trap):
    """DT, Trap: (batch, nheads, seqlen). Returns scale, gamma same shape."""
    trap_sig = jax.nn.sigmoid(Trap)
    dt_shifted = jnp.pad(DT[:, :, 1:], [(0, 0), (0, 0), (0, 1)])
    gamma = DT * trap_sig
    scale = dt_shifted - jnp.diff(gamma, append=0)
    return scale, gamma


def _qk_dot(q: Array, k: Array, q0: Array, k0: Array) -> Array:
    """
    q, k:           (batch, seqlen, nheads_qk, headdim_qk)
    q_bias, k_bias: (nheads, headdim_qk)
    return:         (batch, seqlen, nheads)
    """
    q0 = rearrange(q0, "(h r) d -> h r d", h=q.shape[2])
    k0 = rearrange(k0, "(h r) d -> h r d", h=k.shape[2])
    out = jnp.einsum("bshd,hrd->bshr", q, k0) + jnp.einsum("bshd,hrd->bshr", k, q0)
    out += jnp.einsum("bshd,bshd->bsh", q, k)[..., None]
    out += (q0 * k0).sum(-1)
    return rearrange(out, "b s h r -> b s (h r)")


def _mamba3_scan(Q, K, V, ADT, angles, scale, gamma, D, Z, Q_bias, K_bias, chunk_size):
    """Chunked SSM: parallel intra-chunk precompute + lean sequential scan.

    Q, K:           (batch, seqlen, nheads_qk, headdim_qk)
    V:              (batch, seqlen, nheads, headdim_v)
    ADT:            (batch, nheads, seqlen)  fp32
    angles:         (batch, seqlen, nheads, headdim_angles)
    scale, gamma:   (batch, nheads, seqlen)
    D:              (nheads,)
    Z:              (batch, seqlen, nheads, headdim_v) or None
    Q_bias, K_bias: (nheads, headdim_qk)
    """
    batch = Q.shape[0]
    gqa_ratio = V.shape[2] // Q.shape[2]

    def _to_chunks(x_blhd: Array) -> Array:
        return rearrange(x_blhd, "b (nc cs) h d -> nc (b h) cs d", cs=chunk_size)

    def _to_chunks_bhl(x: Array) -> Array:
        return rearrange(x, "b h (nc cs) -> nc (b h) cs", cs=chunk_size)

    v_s = _to_chunks(V * scale.mT[..., None])
    da_cumsum = _to_chunks_bhl(ADT).cumsum(axis=-1)
    angles = _to_chunks(angles)
    q_s = _to_chunks(Q)
    k_s = _to_chunks(K)

    q_bias_bh = repeat(Q_bias, "h d -> (b h) 1 d", b=batch)
    k_bias_bh = repeat(K_bias, "h d -> (b h) 1 d", b=batch)

    def _gqa_with_rotary(vec: Array, bias: Array, cos: Array, sin: Array) -> Array:
        # vec: (bhq cs d), bias: (bh d)
        vec = repeat(vec, "bhq cs d -> (bhq r) cs d", r=gqa_ratio) + bias
        return _apply_rotary_batched(vec, cos, sin)

    def _intra_chunk_pre(k, v, da_cs, ang):
        k = _gqa_with_rotary(k, k_bias_bh, jnp.cos(ang), jnp.sin(ang))
        # State update: V_scaled^T @ K → (BH, hv, hqk), precomputed
        weight = jnp.exp(da_cs[:, -1:] - da_cs).astype(jnp.bfloat16)
        return v.mT * weight[:, None, :] @ k  # (BH, hv, hqk)

    def _intra_chunk_post(q, k, v, da_cs, prev_state, ang):
        cos_a = jnp.cos(ang)
        sin_a = jnp.sin(ang)
        q = _gqa_with_rotary(q, q_bias_bh, cos_a, sin_a)
        k = _gqa_with_rotary(k, k_bias_bh, cos_a, sin_a)
        # Intra-chunk causal attention (clipping is done to prevent inf grads)
        mask = jnp.exp((da_cs[..., None] - da_cs[:, None]).clip(max=0))
        s = q @ k.mT * jnp.tril(mask.astype(jnp.bfloat16), k=-1)
        exp_cs = jnp.exp(da_cs).astype(jnp.bfloat16)[..., None]
        return s @ v + q @ prev_state.mT * exp_cs

    state_update = jax.vmap(jax.checkpoint(_intra_chunk_pre))(
        k_s, v_s, da_cumsum, angles
    )
    exp_cs = jnp.exp(da_cumsum[..., -1]).astype(jnp.bfloat16)
    states = trace(state_update[:-1], exp_cs[:-1, :, None, None])  # Inter-chunk
    out = jax.vmap(jax.checkpoint(_intra_chunk_post))(
        q_s, k_s, v_s, da_cumsum, states, angles
    )
    out = rearrange(out, "nc (b h) cs d -> b (nc cs) h d", b=batch)
    out += (D + gamma.mT * _qk_dot(Q, K, Q_bias, K_bias))[..., None] * V
    return out if Z is None else out * jax.nn.silu(Z)


def mamba3_siso_combined(
    Q: Array,
    K: Array,
    V: Array,
    ADT: Array,
    DT: Array,
    Trap: Array,
    Q_bias: Array,
    K_bias: Array,
    Angles: Array,
    D: Array | None = None,
    Z: Array | None = None,
    chunk_size: int = 64,
) -> Array:
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
    _, seqlen, nheads, _ = V.shape

    # Pad seqlen to multiple of chunk_size
    pad_len = (-seqlen) % chunk_size

    def _pad_seq(x, ax=1):
        pad_widths = [(0, 0)] * x.ndim
        pad_widths[ax] = (0, pad_len)
        return jnp.pad(x, pad_widths) if pad_len else x

    Q = _pad_seq(Q).astype(jnp.bfloat16)
    K = _pad_seq(K).astype(jnp.bfloat16)
    V = _pad_seq(V).astype(jnp.bfloat16)
    Angles = _pad_seq(Angles).astype(jnp.bfloat16)
    Q_bias = Q_bias.astype(jnp.bfloat16)
    K_bias = K_bias.astype(jnp.bfloat16)
    ADT = _pad_seq(ADT, ax=2).astype(jnp.float32)
    DT = _pad_seq(DT, ax=2).astype(jnp.bfloat16)
    Trap = _pad_seq(Trap, ax=2).astype(jnp.bfloat16)
    if Z is not None:
        Z = _pad_seq(Z).astype(jnp.bfloat16)

    scale, gamma = _compute_scale_gamma(DT, Trap)
    angle_cumsum = angle_dt_cumsum(Angles, DT)
    D = jnp.zeros(nheads, dtype=jnp.bfloat16) if D is None else D.astype(jnp.bfloat16)

    out = _mamba3_scan(
        Q, K, V, ADT, angle_cumsum, scale, gamma, D, Z, Q_bias, K_bias, chunk_size
    )
    return out[:, :seqlen]

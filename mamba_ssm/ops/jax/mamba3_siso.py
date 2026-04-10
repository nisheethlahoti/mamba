"""Mamba-3 SISO in pure JAX"""

import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from jax import Array, lax


def _angle_dt_scan_body(state, chunk):
    """angle_dt_cumsum. state: (BH, dim), chunk: (BH, chunk_size, dim)"""
    out = (chunk.cumsum(axis=1) + state[:, None, :]) % (2 * jnp.pi)
    new_state = (state + chunk.sum(axis=1)) % (2 * jnp.pi)
    return new_state, out


def angle_dt_cumsum(angles: Array, dt: Array, chunk_size: int = 64) -> Array:
    """Compute cumsum(tanh(angles) * pi * dt) mod 2pi, chunked.

    Args:
        angles:     (batch, seqlen, nheads, dim)
        dt:         (batch, nheads, seqlen)
    Returns:
        out:         (batch, seqlen, nheads, dim)
    """
    batch, _, nheads, dim = angles.shape

    vals = jnp.tanh(angles.astype(jnp.float32)) * jnp.pi
    vals = rearrange(vals, "b (nc cs) h d -> b h nc cs d", cs=chunk_size)
    dt_r = rearrange(dt, "b h (nc cs) -> b h nc cs 1", cs=chunk_size)
    vals = rearrange(vals * dt_r, "b h nc cs d -> nc (b h) cs d")

    _, out = lax.scan(_angle_dt_scan_body, jnp.zeros((batch * nheads, dim)), vals)
    out = rearrange(out, "nc (b h) cs d -> b (nc cs) h d", b=batch, h=nheads)
    return out.astype(angles.dtype)


def _apply_rotary_batched(x, cos, sin):
    """Apply rotary embedding, batched.

    x:   (BH, chunk_size, headdim_qk)
    cos: (BH, chunk_size, n_rot)
    sin: (BH, chunk_size, n_rot)
    """
    n_rot = cos.shape[-1]
    x0, x1 = rearrange(x, "bh l (p two) -> two bh l p", two=2)

    ro0 = x0[..., :n_rot] * cos - x1[..., :n_rot] * sin
    ro1 = x0[..., :n_rot] * sin + x1[..., :n_rot] * cos

    out0 = jnp.concatenate([ro0, x0[..., n_rot:]], axis=-1)
    out1 = jnp.concatenate([ro1, x1[..., n_rot:]], axis=-1)
    return rearrange([out0, out1], "two bh l p -> bh l (p two)")


def _compute_scale_gamma(DT, Trap):
    """DT, Trap: (batch, nheads, seqlen). Returns scale, gamma same shape."""
    dt_f32 = DT.astype(jnp.float32)
    trap_sig = jax.nn.sigmoid(Trap.astype(jnp.float32))
    # Shifted versions: value at t+1, zero-padded at the end
    dt_shifted = jnp.pad(dt_f32[:, :, 1:], [(0, 0), (0, 0), (0, 1)])
    trap_sig_shifted = jnp.pad(trap_sig[:, :, 1:], [(0, 0), (0, 0), (0, 1)])
    gamma = dt_f32 * trap_sig
    scale = dt_shifted * (1 - trap_sig_shifted) + gamma
    return scale, gamma


def _mamba3_scan(Q, K, V, ADT, angles, scale, gamma, D, Z, Q_bias, K_bias, chunk_size):
    """Chunked SSM. All preprocessing done per-chunk inside scan body.

    Q, K:           (batch, seqlen, nheads_qk, headdim_qk)     bf16
    V:              (batch, seqlen, nheads, headdim_v)         bf16
    ADT:            (batch, nheads, seqlen)                    f32
    angles:         (batch, seqlen, nheads, headdim_angles)    bf16
    scale, gamma:   (batch, nheads, seqlen)                    f32
    D:              (nheads,)                                  bf16
    Z:              (batch, seqlen, nheads, headdim_v) or None bf16
    Q_bias, K_bias: (nheads, headdim_qk)                       bf16
    """
    batch, _, nheads_qk, headdim_qk = Q.shape
    nheads = V.shape[2]
    headdim_v = V.shape[3]
    BH = batch * nheads
    gqa_ratio = nheads // nheads_qk

    def _to_chunks(x_blhd):
        return rearrange(x_blhd, "b (nc cs) h d -> nc (b h) cs d", cs=chunk_size)

    def _to_chunks_bhl(x):
        return rearrange(x, "b h (nc cs) -> nc (b h) cs", cs=chunk_size)

    v_s = _to_chunks(V)
    adt_s = _to_chunks_bhl(ADT.astype(jnp.float32))
    scale_s = _to_chunks_bhl(scale.astype(jnp.bfloat16))
    gamma_s = _to_chunks_bhl(gamma.astype(jnp.bfloat16))
    cos_s = _to_chunks(jnp.cos(angles))
    sin_s = _to_chunks(jnp.sin(angles))
    q_s = _to_chunks(Q)
    k_s = _to_chunks(K)

    d_bh = repeat(D.astype(jnp.float32), "h -> (b h)", b=batch)

    state_init = jnp.zeros((BH, headdim_v, headdim_qk), dtype=jnp.float32)
    causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=-1)

    def _scan_body(ssm_state, inputs):
        """One chunk, all batch*heads in parallel via batched matmul.

        ssm_state: (BH, headdim_v, headdim_qk)
        """
        q, k, v, adt, sc, gm, cos_a, sin_a = inputs
        # q, k: (BHq, chunk_size, hqk)
        # v:    (BH, chunk_size, hv)
        # adt, sc, gm: (BH, chunk_size)
        # cos_a, sin_a: (BH, chunk_size, ha)

        # GQA expand Q, K per-chunk
        if gqa_ratio > 1:
            q = repeat(q, "bhq cs d -> (bhq r) cs d", r=gqa_ratio)
            k = repeat(k, "bhq cs d -> (bhq r) cs d", r=gqa_ratio)

        # Apply bias (after GQA expansion)
        q = q + repeat(Q_bias, "h d -> (b h) 1 d", b=batch)
        k = k + repeat(K_bias, "h d -> (b h) 1 d", b=batch)

        # QK dot (before rotary)
        qk_dot = (q * k).sum(axis=-1) * gm  # (BH, chunk_size)

        # Rotary (precomputed cos/sin)
        q = _apply_rotary_batched(q, cos_a, sin_a)
        k = _apply_rotary_batched(k, cos_a, sin_a)

        # Scale K
        k = k * sc.astype(jnp.bfloat16)[:, :, None]

        # --- SSM ---
        da_cs = adt.cumsum(axis=-1)
        da_cs_last = da_cs[:, -1]

        # Inter-chunk: (BH, L, hqk) @ (BH, hqk, hv) -> (BH, L, hv)
        acc_o = q @ ssm_state.astype(jnp.bfloat16).transpose(0, 2, 1)
        acc_o = acc_o * jnp.exp(da_cs)[:, :, None]

        # Intra-chunk: (BH, L, hqk) @ (BH, hqk, L) -> (BH, L, L)
        s: Array = q @ k.transpose(0, 2, 1)
        s = s * jnp.exp(jnp.minimum(da_cs[:, :, None] - da_cs[:, None, :], 0.0))
        s = jnp.where(causal_mask[None, :, :], s, 0.0)
        acc_o = acc_o + s.astype(jnp.bfloat16) @ v

        # D-skip + QK diagonal
        acc_o = acc_o + (d_bh[:, None] + qk_dot)[:, :, None] * v

        # State update
        da_cs_rev = da_cs_last[:, None] - da_cs
        v_scaled = v * jnp.exp(da_cs_rev)[:, :, None]
        new_state = ssm_state * jnp.exp(da_cs_last)[:, None, None]
        # (BH, hv, L) @ (BH, L, hqk) -> (BH, hv, hqk)
        new_state += v_scaled.transpose(0, 2, 1) @ k

        return new_state, acc_o.astype(jnp.bfloat16)

    arrs = q_s, k_s, v_s, adt_s, scale_s, gamma_s, cos_s, sin_s
    _, out_s = lax.scan(jax.checkpoint(_scan_body), state_init, arrs)
    out = rearrange(out_s, "nc (b h) cs d -> b (nc cs) h d", b=batch, h=nheads)

    if Z is not None:
        out = out * jax.nn.silu(Z.astype(jnp.float32))
    return out


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
        return jnp.pad(x, pad_widths)

    Q = _pad_seq(Q).astype(jnp.bfloat16)
    K = _pad_seq(K).astype(jnp.bfloat16)
    V = _pad_seq(V).astype(jnp.bfloat16)
    Angles = _pad_seq(Angles).astype(jnp.bfloat16)
    Q_bias = Q_bias.astype(jnp.bfloat16)
    K_bias = K_bias.astype(jnp.bfloat16)
    ADT = _pad_seq(ADT, ax=2)
    DT = _pad_seq(DT, ax=2)
    Trap = _pad_seq(Trap, ax=2)
    if Z is not None:
        Z = _pad_seq(Z).astype(jnp.bfloat16)

    scale, gamma = _compute_scale_gamma(DT, Trap)
    angle_cumsum = angle_dt_cumsum(Angles, DT, chunk_size=chunk_size)
    D = jnp.zeros(nheads, dtype=jnp.float32) if D is None else D

    out = _mamba3_scan(
        Q, K, V, ADT, angle_cumsum, scale, gamma, D, Z, Q_bias, K_bias, chunk_size
    )
    return out[:, :seqlen]

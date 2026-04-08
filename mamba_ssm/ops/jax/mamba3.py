from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


Array = jax.Array


class Mamba3SISOState(NamedTuple):
    angle_state: Array
    ssm_state: Array
    k_state: Array
    v_state: Array


def _expand_gqa(x: Array, nheads: int, *, head_axis: int) -> Array:
    if x.shape[head_axis] == nheads:
        return x
    repeats = nheads // x.shape[head_axis]
    return jnp.repeat(x, repeats, axis=head_axis)


def _pad_rotary(cos: Array, sin: Array, target_pairs: int) -> tuple[Array, Array]:
    pad = target_pairs - cos.shape[-1]
    if pad <= 0:
        return cos, sin
    pad_cfg = [(0, 0)] * (cos.ndim - 1) + [(0, pad)]
    cos = jnp.pad(cos, pad_cfg, constant_values=1.0)
    sin = jnp.pad(sin, pad_cfg, constant_values=0.0)
    return cos, sin


def _apply_rotary(x: Array, cos: Array, sin: Array) -> Array:
    pairs = x.shape[-1] // 2
    cos, sin = _pad_rotary(cos, sin, pairs)
    x_pairs = x.reshape(*x.shape[:-1], pairs, 2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return jnp.stack((y0, y1), axis=-1).reshape(x.shape)


def _segsum(x: Array) -> Array:
    chunk = x.shape[-1]
    i = jnp.arange(chunk)[:, None]
    j = jnp.arange(chunk)[None, :]
    mask = i >= j
    prefix = jnp.cumsum(x, axis=-1)
    seg = prefix[..., :, None] - prefix[..., None, :]
    neg_inf = jnp.array(-jnp.inf, dtype=x.dtype)
    return jnp.where(mask, seg, neg_inf)


def _mod_2pi(x: Array) -> Array:
    two_pi = jnp.array(2.0 * math.pi, dtype=x.dtype)
    return x - two_pi * jnp.floor(x / two_pi)


def _prepare_common(
    q: Array,
    k: Array,
    v: Array,
    adt: Array,
    dt: Array,
    trap: Array,
    q_bias: Array,
    k_bias: Array,
    angles: Array,
    initial_angle_state: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    nheads = v.shape[2]
    q = _expand_gqa(q, nheads, head_axis=2)
    k = _expand_gqa(k, nheads, head_axis=2)
    q_pre = q + q_bias[None, None, :, :]
    k_pre = k + k_bias[None, None, :, :]
    trap_sigmoid = jax.nn.sigmoid(trap)
    shifted_dt = jnp.pad(dt[:, :, 1:], ((0, 0), (0, 0), (0, 1)))
    shifted_trap = jnp.pad(trap_sigmoid[:, :, 1:], ((0, 0), (0, 0), (0, 1)))
    shifted_gamma = shifted_dt * (1.0 - shifted_trap)
    scale = dt * trap_sigmoid + shifted_gamma
    raw_angles = jnp.tanh(angles.astype(jnp.float32)) * jnp.float32(math.pi)
    angles_cumsum = jnp.cumsum(raw_angles * dt.transpose(0, 2, 1)[..., None], axis=1)
    if initial_angle_state is not None:
        angles_cumsum = angles_cumsum + initial_angle_state[:, None, :, :]
    angles_cumsum = _mod_2pi(angles_cumsum)
    cos = jnp.cos(angles_cumsum).astype(q_pre.dtype)
    sin = jnp.sin(angles_cumsum).astype(q_pre.dtype)
    q_rot = _apply_rotary(q_pre, cos, sin)
    k_rot = _apply_rotary(k_pre, cos, sin)
    qk_dot = jnp.sum(q_pre * k_pre, axis=-1) * shifted_gamma.transpose(0, 2, 1).astype(q_pre.dtype)
    k_scaled = k_rot * scale.transpose(0, 2, 1)[..., None].astype(k_rot.dtype)
    return q_rot, k_rot, k_scaled, qk_dot, angles_cumsum, trap_sigmoid, scale


def _prepare_forward_metadata(
    q: Array,
    k: Array,
    v: Array,
    dt: Array,
    trap: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    nheads = v.shape[2]
    q = _expand_gqa(q, nheads, head_axis=2)
    k = _expand_gqa(k, nheads, head_axis=2)
    trap_sigmoid = jax.nn.sigmoid(trap)
    shifted_dt = jnp.pad(dt[:, :, 1:], ((0, 0), (0, 0), (0, 1)))
    shifted_trap = jnp.pad(trap_sigmoid[:, :, 1:], ((0, 0), (0, 0), (0, 1)))
    shifted_gamma = shifted_dt * (1.0 - shifted_trap)
    scale = dt * trap_sigmoid + shifted_gamma
    return q, k, trap_sigmoid, shifted_gamma, scale


def _step_common(
    q: Array,
    k: Array,
    adt: Array,
    dt: Array,
    trap: Array,
    q_bias: Array,
    k_bias: Array,
    angles: Array,
    state: Mamba3SISOState,
) -> tuple[Array, Array, Array, Array, Array]:
    q = _expand_gqa(q, state.k_state.shape[1], head_axis=1)
    k = _expand_gqa(k, state.k_state.shape[1], head_axis=1)
    q_pre = q + q_bias[None, :, :]
    k_pre = k + k_bias[None, :, :]
    angle_state = _mod_2pi(
        state.angle_state + jnp.tanh(angles.astype(jnp.float32)) * jnp.float32(math.pi) * dt[..., None]
    )
    cos = jnp.cos(angle_state).astype(q_pre.dtype)
    sin = jnp.sin(angle_state).astype(q_pre.dtype)
    q_rot = _apply_rotary(q_pre, cos, sin)
    k_rot = _apply_rotary(k_pre, cos, sin)
    trap_sigmoid = jax.nn.sigmoid(trap)
    alpha = jnp.exp(adt).astype(jnp.float32)
    beta = alpha * dt * (1.0 - trap_sigmoid)
    gamma = dt * trap_sigmoid
    return angle_state, q_rot, k_rot, alpha, beta, gamma


def mamba3_siso_step_jax(
    q: Array,
    k: Array,
    v: Array,
    adt: Array,
    dt: Array,
    trap: Array,
    q_bias: Array,
    k_bias: Array,
    angles: Array,
    state: Mamba3SISOState,
    *,
    d: Array | None = None,
    z: Array | None = None,
) -> tuple[Array, Mamba3SISOState]:
    angle_state, q_rot, k_rot, alpha, beta, gamma = _step_common(
        q, k, adt, dt, trap, q_bias, k_bias, angles, state
    )
    ssm_state = (
        state.ssm_state * alpha[:, :, None, None]
        + beta[:, :, None, None] * state.v_state[:, :, :, None] * state.k_state[:, :, None, :]
        + gamma[:, :, None, None] * v[:, :, :, None] * k_rot[:, :, None, :]
    )
    out = jnp.einsum("bhvq,bhq->bhv", ssm_state, q_rot.astype(ssm_state.dtype)).astype(v.dtype)
    if d is not None:
        out = out + d[None, :, None].astype(out.dtype) * v
    if z is not None:
        out = out * (z * jax.nn.sigmoid(z))
    return out, Mamba3SISOState(angle_state, ssm_state, k_rot, v)


def _step_pallas_kernel(
    q_ref,
    k_ref,
    v_ref,
    alpha_ref,
    beta_ref,
    gamma_ref,
    ssm_ref,
    prev_k_ref,
    prev_v_ref,
    out_ref,
    ssm_out_ref,
    k_out_ref,
    v_out_ref,
):
    q = q_ref[0, 0, :].astype(jnp.float32)
    k = k_ref[0, 0, :].astype(jnp.float32)
    v = v_ref[0, 0, :].astype(jnp.float32)
    prev_k = prev_k_ref[0, 0, :].astype(jnp.float32)
    prev_v = prev_v_ref[0, 0, :].astype(jnp.float32)
    ssm = ssm_ref[0, 0, :, :].astype(jnp.float32)
    alpha = alpha_ref[0, 0].astype(jnp.float32)
    beta = beta_ref[0, 0].astype(jnp.float32)
    gamma = gamma_ref[0, 0].astype(jnp.float32)

    ssm = (
        alpha * ssm
        + beta * prev_v[:, None] * prev_k[None, :]
        + gamma * v[:, None] * k[None, :]
    )
    out = jnp.sum(ssm * q[None, :], axis=1)

    out_ref[0, 0, :] = out.astype(out_ref.dtype)
    ssm_out_ref[0, 0, :, :] = ssm
    k_out_ref[0, 0, :] = k.astype(k_out_ref.dtype)
    v_out_ref[0, 0, :] = v.astype(v_out_ref.dtype)


def mamba3_siso_step_pallas(
    q: Array,
    k: Array,
    v: Array,
    adt: Array,
    dt: Array,
    trap: Array,
    q_bias: Array,
    k_bias: Array,
    angles: Array,
    state: Mamba3SISOState,
    *,
    d: Array | None = None,
    z: Array | None = None,
    interpret: bool = False,
) -> tuple[Array, Mamba3SISOState]:
    angle_state, q_rot, k_rot, alpha, beta, gamma = _step_common(
        q, k, adt, dt, trap, q_bias, k_bias, angles, state
    )
    batch, nheads, headdim_v = v.shape
    headdim_qk = q_rot.shape[-1]
    blockspec_bhq = pl.BlockSpec((1, 1, headdim_qk), lambda b, h: (b, h, 0))
    blockspec_bhv = pl.BlockSpec((1, 1, headdim_v), lambda b, h: (b, h, 0))
    blockspec_bh = pl.BlockSpec((1, 1), lambda b, h: (b, h))
    blockspec_bhvq = pl.BlockSpec((1, 1, headdim_v, headdim_qk), lambda b, h: (b, h, 0, 0))
    call = pl.pallas_call(
        _step_pallas_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((batch, nheads, headdim_v), v.dtype),
            jax.ShapeDtypeStruct((batch, nheads, headdim_v, headdim_qk), jnp.float32),
            jax.ShapeDtypeStruct((batch, nheads, headdim_qk), k_rot.dtype),
            jax.ShapeDtypeStruct((batch, nheads, headdim_v), v.dtype),
        ),
        grid=(batch, nheads),
        in_specs=(
            blockspec_bhq,
            blockspec_bhq,
            blockspec_bhv,
            blockspec_bh,
            blockspec_bh,
            blockspec_bh,
            blockspec_bhvq,
            blockspec_bhq,
            blockspec_bhv,
        ),
        out_specs=(
            blockspec_bhv,
            blockspec_bhvq,
            blockspec_bhq,
            blockspec_bhv,
        ),
        interpret=interpret,
        name="mamba3_siso_step",
    )
    out, ssm_state, k_state, v_state = call(
        q_rot,
        k_rot,
        v,
        alpha,
        beta,
        gamma,
        state.ssm_state,
        state.k_state,
        state.v_state,
    )
    if d is not None:
        out = out + d[None, :, None].astype(out.dtype) * v
    if z is not None:
        out = out * (z * jax.nn.sigmoid(z))
    return out, Mamba3SISOState(angle_state, ssm_state, k_state, v_state)


def _chunk_scan(
    q: Array,
    k: Array,
    v: Array,
    adt: Array,
    dt: Array,
    shifted_gamma: Array,
    scale: Array,
    angles: Array,
    valid: Array,
    initial_angle_state: Array,
    initial_acc_state: Array,
    initial_k_state: Array,
    initial_v_state: Array,
    q_bias: Array,
    k_bias: Array,
    *,
    chunk_size: int,
    d: Array | None,
    z: Array | None,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    batch, seqlen, nheads, headdim_qk = q.shape
    headdim_v = v.shape[-1]
    nchunks = seqlen // chunk_size

    def chunk_view(x: Array) -> Array:
        shape = (batch, nchunks, chunk_size) + x.shape[2:]
        return x.reshape(shape)

    q_chunks = chunk_view(q)
    k_chunks = chunk_view(k)
    v_chunks = chunk_view(v)
    adt_chunks = adt.reshape(batch, nheads, nchunks, chunk_size).transpose(0, 2, 1, 3)
    dt_chunks = dt.reshape(batch, nheads, nchunks, chunk_size).transpose(0, 2, 1, 3)
    shifted_gamma_chunks = shifted_gamma.reshape(batch, nheads, nchunks, chunk_size).transpose(0, 2, 1, 3)
    scale_chunks = scale.reshape(batch, nheads, nchunks, chunk_size).transpose(0, 2, 1, 3)
    angles_chunks = chunk_view(angles)
    valid_chunks = valid.reshape(batch, nchunks, chunk_size)
    z_chunks = chunk_view(z) if z is not None else None

    def body(
        carry: tuple[Array, Array, Array, Array],
        xs: tuple[Array, ...],
    ) -> tuple[tuple[Array, Array, Array, Array], Array]:
        angle_state, acc_state, _, _ = carry
        q_chunk, k_chunk, v_chunk, adt_chunk, dt_chunk, shifted_gamma_chunk, scale_chunk, angles_chunk, valid_chunk = xs[:9]
        z_chunk = xs[9] if z_chunks is not None else None
        valid_bt = valid_chunk[..., None, None]
        valid_bh = valid_chunk[:, None, :]
        q_pre = jnp.where(valid_bt, q_chunk + q_bias[None, None, :, :], 0)
        k_pre = jnp.where(valid_bt, k_chunk + k_bias[None, None, :, :], 0)
        v_chunk = jnp.where(valid_bt, v_chunk, 0)
        adt_chunk = jnp.where(valid_bh, adt_chunk, 0)
        dt_chunk = jnp.where(valid_bh, dt_chunk, 0)
        shifted_gamma_chunk = jnp.where(valid_bh, shifted_gamma_chunk, 0)
        scale_chunk = jnp.where(valid_bh, scale_chunk, 0)
        angles_chunk = jnp.where(valid_bt, angles_chunk, 0)
        if z_chunk is not None:
            z_chunk = jnp.where(valid_bt, z_chunk, 0)
        angles_scaled = jnp.tanh(angles_chunk.astype(jnp.float32)) * jnp.float32(math.pi)
        angles_scaled = angles_scaled * dt_chunk.transpose(0, 2, 1)[..., None]
        angles_cumsum = _mod_2pi(jnp.cumsum(angles_scaled, axis=1) + angle_state[:, None, :, :])
        cos = jnp.cos(angles_cumsum).astype(q_pre.dtype)
        sin = jnp.sin(angles_cumsum).astype(q_pre.dtype)
        q_rot = _apply_rotary(q_pre, cos, sin)
        k_rot = _apply_rotary(k_pre, cos, sin)
        qk_dot_chunk = jnp.sum(k_pre * q_pre, axis=-1) * shifted_gamma_chunk.transpose(0, 2, 1).astype(q_pre.dtype)
        k_scaled_chunk = k_rot * scale_chunk.transpose(0, 2, 1)[..., None].astype(k_rot.dtype)
        local_decay = jnp.exp(_segsum(adt_chunk))
        qk = jnp.einsum("bthq,bshq->bhts", q_rot, k_scaled_chunk)
        local_out = jnp.einsum("bhts,bshv->bthv", qk * local_decay, v_chunk)
        da_cs = jnp.exp(jnp.cumsum(adt_chunk, axis=-1))
        carry_out = jnp.einsum("bhvq,bthq,bht->bthv", acc_state, q_rot, da_cs)
        out = local_out + carry_out.astype(local_out.dtype)
        if d is not None:
            out = out + d[None, None, :, None].astype(out.dtype) * v_chunk
        out = out - v_chunk * qk_dot_chunk[..., None]
        if z_chunk is not None:
            out = out * (z_chunk * jax.nn.sigmoid(z_chunk))
        chunk_sum = jnp.sum(adt_chunk, axis=-1)
        da_cs_last = jnp.exp(chunk_sum)
        da_cs_rev = jnp.exp(chunk_sum[..., None] - jnp.cumsum(adt_chunk, axis=-1))
        v_scaled = v_chunk * da_cs_rev.transpose(0, 2, 1)[..., None].astype(v_chunk.dtype)
        next_carry = (
            acc_state * da_cs_last[..., None, None]
            + jnp.einsum("bthq,bthv->bhvq", k_scaled_chunk, v_scaled)
        )
        last_valid_idx = jnp.maximum(jnp.sum(valid_chunk, axis=1) - 1, 0).astype(jnp.int32)
        gather_k = jnp.take_along_axis(
            k_rot,
            last_valid_idx[:, None, None, None],
            axis=1,
        )[:, 0]
        gather_v = jnp.take_along_axis(
            v_chunk,
            last_valid_idx[:, None, None, None],
            axis=1,
        )[:, 0]
        has_valid = (jnp.sum(valid_chunk, axis=1) > 0)[:, None, None]
        next_k_state = jnp.where(has_valid, gather_k, carry[2])
        next_v_state = jnp.where(has_valid, gather_v, carry[3])
        next_state = (
            angles_cumsum[:, -1],
            next_carry.astype(jnp.float32),
            next_k_state,
            next_v_state,
        )
        return next_state, out

    scan_init = (
        initial_angle_state,
        initial_acc_state,
        initial_k_state,
        initial_v_state,
    )

    scan_inputs = [q_chunks, k_chunks, v_chunks, adt_chunks, dt_chunks, shifted_gamma_chunks, scale_chunks, angles_chunks, valid_chunks]
    if z_chunks is not None:
        scan_inputs.append(z_chunks)
    final_state, outputs = jax.lax.scan(
        body,
        scan_init,
        [x.swapaxes(0, 1) for x in scan_inputs],
    )
    outputs = outputs.swapaxes(0, 1).reshape(batch, seqlen, nheads, headdim_v)
    return outputs, final_state


@jax.jit
def mamba3_siso_reference(
    q: Array,
    k: Array,
    v: Array,
    adt: Array,
    dt: Array,
    trap: Array,
    q_bias: Array,
    k_bias: Array,
    angles: Array,
    *,
    d: Array | None = None,
    z: Array | None = None,
    initial_state: Mamba3SISOState | None = None,
) -> tuple[Array, Mamba3SISOState]:
    batch, seqlen, _, headdim_qk = q.shape
    nheads = v.shape[2]
    angle_dim = angles.shape[-1]
    if initial_state is None:
        initial_state = Mamba3SISOState(
            jnp.zeros((batch, nheads, angle_dim), dtype=jnp.float32),
            jnp.zeros((batch, nheads, v.shape[-1], headdim_qk), dtype=jnp.float32),
            jnp.zeros((batch, nheads, headdim_qk), dtype=q.dtype),
            jnp.zeros((batch, nheads, v.shape[-1]), dtype=v.dtype),
        )

    def body(state: Mamba3SISOState, xs: tuple[Array, ...]) -> tuple[Mamba3SISOState, Array]:
        q_t, k_t, v_t, adt_t, dt_t, trap_t, angles_t = xs[:7]
        z_t = xs[7] if z is not None else None
        out_t, next_state = mamba3_siso_step_jax(
            q_t,
            k_t,
            v_t,
            adt_t,
            dt_t,
            trap_t,
            q_bias,
            k_bias,
            angles_t,
            state,
            d=d,
            z=z_t,
        )
        return next_state, out_t

    scan_inputs = [q, k, v, adt.transpose(0, 2, 1), dt.transpose(0, 2, 1), trap.transpose(0, 2, 1), angles]
    if z is not None:
        scan_inputs.append(z)
    final_state, out = jax.lax.scan(body, initial_state, [x.swapaxes(0, 1) for x in scan_inputs])
    return out.swapaxes(0, 1), final_state


@jax.jit(static_argnames=("chunk_size",))
def mamba3_siso(
    q: Array,
    k: Array,
    v: Array,
    adt: Array,
    dt: Array,
    trap: Array,
    q_bias: Array,
    k_bias: Array,
    angles: Array,
    *,
    chunk_size: int = 64,
    d: Array | None = None,
    z: Array | None = None,
    initial_state: Mamba3SISOState | None = None,
) -> tuple[Array, Mamba3SISOState]:
    batch, seqlen, _, headdim_qk = q.shape
    nheads = v.shape[2]
    angle_dim = angles.shape[-1]
    if initial_state is None:
        initial_state = Mamba3SISOState(
            jnp.zeros((batch, nheads, angle_dim), dtype=jnp.float32),
            jnp.zeros((batch, nheads, v.shape[-1], headdim_qk), dtype=jnp.float32),
            jnp.zeros((batch, nheads, headdim_qk), dtype=q.dtype),
            jnp.zeros((batch, nheads, v.shape[-1]), dtype=v.dtype),
        )

    q, k, trap_sigmoid, shifted_gamma, scale = _prepare_forward_metadata(q, k, v, dt, trap)
    pad = (-seqlen) % chunk_size
    pad_seq = ((0, 0), (0, pad), (0, 0), (0, 0))
    q_pad = jnp.pad(q, pad_seq)
    k_pad = jnp.pad(k, pad_seq)
    v_pad = jnp.pad(v, ((0, 0), (0, pad), (0, 0), (0, 0)))
    adt_pad = jnp.pad(adt, ((0, 0), (0, 0), (0, pad)))
    dt_pad = jnp.pad(dt, ((0, 0), (0, 0), (0, pad)))
    shifted_gamma_pad = jnp.pad(shifted_gamma, ((0, 0), (0, 0), (0, pad)))
    scale_pad = jnp.pad(scale, ((0, 0), (0, 0), (0, pad)))
    angles_pad = jnp.pad(angles, pad_seq)
    valid_pad = jnp.pad(jnp.ones((batch, seqlen), dtype=bool), ((0, 0), (0, pad)), constant_values=False)
    z_pad = jnp.pad(z, ((0, 0), (0, pad), (0, 0), (0, 0))) if z is not None else None

    first_beta = dt[:, :, 0] * (1.0 - trap_sigmoid[:, :, 0])
    initial_acc_state = (
        initial_state.ssm_state
        + first_beta[:, :, None, None]
        * initial_state.v_state[:, :, :, None]
        * initial_state.k_state[:, :, None, :]
    )
    out, final_scan_state = _chunk_scan(
        q_pad,
        k_pad,
        v_pad,
        adt_pad,
        dt_pad,
        shifted_gamma_pad,
        scale_pad,
        angles_pad,
        valid_pad,
        initial_state.angle_state,
        initial_acc_state,
        initial_state.k_state,
        initial_state.v_state,
        q_bias,
        k_bias,
        chunk_size=chunk_size,
        d=d,
        z=z_pad,
    )
    out = out[:, :seqlen]
    final_angle_state, final_ssm_state, final_k_state, final_v_state = final_scan_state
    final_state = Mamba3SISOState(
        final_angle_state,
        final_ssm_state,
        final_k_state,
        final_v_state,
    )
    return out, final_state

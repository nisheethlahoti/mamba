"""Mamba-3 SISO in pure JAX"""

# ruff: noqa: F722
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import Array
from jax._src.lax.control_flow.loops import _interleave
from jaxtyping import Float


def parallel_scan(x: Array, coeff: Array) -> Array:
    """Parallel prefix scan for the linear recurrence s[i+1] = coeff[i] * s[i] + x[i].

    Returns all n+1 states s[0..n] (with s[0] = 0) using a Blelloch-style
    divide-and-conquer over axis 0, giving O(n) work in O(log n) depth.
    """
    if not x.shape[0]:
        return jnp.zeros((1,) + x.shape[1:], x.dtype)
    x0, x1 = x[::2], x[1::2]
    c0, c1 = coeff[::2], coeff[1::2]
    arr = parallel_scan(x0[: x1.shape[0]] * c1 + x1, c0[: c1.shape[0]] * c1)
    return _interleave(arr, arr[: x0.shape[0]] * c0 + x0, axis=0)


def angle_dt_cumsum(
    angles: Float[Array, "b s nh d"], dt: Float[Array, "b nh s"]
) -> Float[Array, "b s nh d"]:
    """Compute cumsum(tanh(angles) * pi * dt) mod 2pi."""
    vals = jnp.tanh(angles.astype(jnp.float32)) * jnp.pi
    vals *= rearrange(dt, "b h s -> b s h 1")
    out = vals.cumsum(axis=1) % (2 * jnp.pi)
    return out.astype(angles.dtype)


def _apply_rotary_batched(
    x: Float[Array, "b s d"], cos: Float[Array, "b s rot"], sin: Float[Array, "b s rot"]
) -> Float[Array, "b s d"]:
    """Apply rotary embedding, batched."""
    n_rot = cos.shape[-1]
    x = rearrange(x, "bh l (p two) -> bh l p two", two=2)
    x0, x1 = jnp.unstack(x, axis=-1)

    ro0 = x0[..., :n_rot] * cos - x1[..., :n_rot] * sin
    ro1 = x0[..., :n_rot] * sin + x1[..., :n_rot] * cos

    out0 = jnp.concatenate([ro0, x0[..., n_rot:]], axis=-1)
    out1 = jnp.concatenate([ro1, x1[..., n_rot:]], axis=-1)
    out = jnp.stack([out0, out1], axis=-1)
    return rearrange(out, "bh l p two -> bh l (p two)")


def _compute_scale_gamma(
    DT: Float[Array, "b nh s"], Trap: Float[Array, "b nh s"]
) -> tuple[Float[Array, "b nh s"], Float[Array, "b nh s"]]:
    """Compute trapezoidal scale and gamma from DT and Trap."""
    trap_sig = jax.nn.sigmoid(Trap)
    dt_shifted = jnp.pad(DT[:, :, 1:], [(0, 0), (0, 0), (0, 1)])
    gamma = DT * trap_sig
    scale = dt_shifted - jnp.diff(gamma, append=0)
    return scale, gamma


def _mamba3_fn(
    Q: Float[Array, "seqlen dqk"],
    K: Float[Array, "seqlen dqk"],
    V: Float[Array, "seqlen gqa dv"],
    ADT: Float[Array, "gqa seqlen"],
    angle: Float[Array, "seqlen gqa hdangles"],
    scale: Float[Array, "gqa seqlen"],
    gamma: Float[Array, "gqa seqlen"],
    D: Float[Array, "gqa"],  # noqa: F821
    Q_bias: Float[Array, "gqa dqk"],
    K_bias: Float[Array, "gqa dqk"],
    chunk_size: int,
) -> Float[Array, "seqlen gqa dv"]:
    """Chunked SSM for a single QK-head."""

    def _to_chunks(x: Array, axis: int) -> tuple[int, Array]:
        """Split `axis` into two axes, the second with size `chunk_size`. Return
        tuple of (axis, chunked_array)"""
        y = x.reshape(*x.shape[:axis], -1, chunk_size, *x.shape[axis + 1 :])
        return axis, y

    v_s = _to_chunks(V * scale.T[..., None], 0)
    adt_ax, adt_arr = _to_chunks(ADT, 1)
    adt_cumsum = adt_ax, adt_arr.cumsum(axis=-1)
    angle_s = _to_chunks(angle, 0)
    q_s = _to_chunks(Q, 0)
    k_s = _to_chunks(K, 0)

    def intra_chunk_pre(
        k: Float[Array, "cs dqk"],
        v: Float[Array, "cs gqa dv"],
        adt_cs: Float[Array, "gqa cs"],
        ang: Float[Array, "cs gqa rot"],
    ) -> Float[Array, "gqa dv dqk"]:
        ang = ang.transpose(1, 0, 2)
        k = _apply_rotary_batched(k + K_bias[:, None], jnp.cos(ang), jnp.sin(ang))
        weight = jnp.exp(adt_cs[:, -1:] - adt_cs).astype(v.dtype)
        return v.transpose(1, 2, 0) * weight[:, None, :] @ k

    def intra_chunk_post(
        q: Float[Array, "cs dqk"],
        k: Float[Array, "cs dqk"],
        v: Float[Array, "cs gqa dv"],
        adt_cs: Float[Array, "gqa cs"],
        prev_state: Float[Array, "gqa dv dqk"],
        ang: Float[Array, "cs gqa rot"],
    ) -> Float[Array, "gqa cs dv"]:
        ang = ang.transpose(1, 0, 2)
        cos_a = jnp.cos(ang)
        sin_a = jnp.sin(ang)
        q = _apply_rotary_batched(q + Q_bias[:, None], cos_a, sin_a)
        k = _apply_rotary_batched(k + K_bias[:, None], cos_a, sin_a)
        # Intra-chunk causal attention (clipping is done to prevent inf grads)
        # Mask and s have shape (gqa_ratio, chunk, chunk)
        mask = jnp.exp((adt_cs[..., None] - adt_cs[:, None]).clip(max=0))
        s = q @ k.mT * jnp.tril(mask.astype(v.dtype), k=-1)
        exp_cs = jnp.exp(adt_cs).astype(v.dtype)[..., None]
        return s @ v.transpose(1, 0, 2) + q @ prev_state.mT * exp_cs

    def chunked(fn, *args) -> Array:
        fn = jax.vmap(jax.checkpoint(fn), in_axes=[a[0] for a in args])
        return fn(*(a[1] for a in args))

    state_update = chunked(intra_chunk_pre, k_s, v_s, adt_cumsum, angle_s)
    exp_cs = jnp.exp(adt_cumsum[1][..., -1].T).astype(V.dtype)
    states = parallel_scan(state_update[:-1], exp_cs[:-1, :, None, None])  # Inter-chunk
    out = chunked(intra_chunk_post, q_s, k_s, v_s, adt_cumsum, (0, states), angle_s)
    out = rearrange(out, "nc r cs d -> (nc cs) r d")
    qk_dot = jnp.vecdot(Q, K)[..., None]  # Ultimate shape wanted = (seqlen, gqa_ratio)
    qk_dot += Q @ K_bias.T + K @ Q_bias.T + jnp.vecdot(Q_bias, K_bias)
    return out + (D + gamma.T * qk_dot)[..., None] * V


def mamba3_siso_combined(
    Q: Float[Array, "batch seqlen nheads_qk dqk"],
    K: Float[Array, "batch seqlen nheads_qk dqk"],
    V: Float[Array, "batch seqlen nheads dv"],
    ADT: Float[Array, "batch nheads seqlen"],
    DT: Float[Array, "batch nheads seqlen"],
    Trap: Float[Array, "batch nheads seqlen"],
    Q_bias: Float[Array, "nheads dqk"],
    K_bias: Float[Array, "nheads dqk"],
    Angles: Float[Array, "batch seqlen nheads hdangles"],
    D: Float[Array, "nheads"] | None = None,  # noqa: F821
    Z: Float[Array, "batch seqlen nheads dv"] | None = None,
    chunk_size: int = 64,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Float[Array, "batch seqlen nheads dv"]:
    """Mamba-3 SISO forward pass in pure JAX.
    chunk_size must be passed via functools.partial for jit.
    """
    seqlen = V.shape[1]
    nheads_qk = Q.shape[2]
    pad_len = (-seqlen) % chunk_size

    def pad(x, axis=1, dt=dtype):
        """Pad sequence axis to a multiple of chunk_size and cast."""
        pads = [(0, 0)] * x.ndim
        pads[axis] = (0, pad_len)
        return jnp.pad(x, pads).astype(dt)

    def split(x, ax):
        """Split axis `ax` from nheads into (nheads_qk, gqa_ratio)."""
        return x.reshape(*x.shape[:ax], nheads_qk, -1, *x.shape[ax + 1 :])

    Q = pad(Q)
    K = pad(K)
    V = split(pad(V), 2)
    Angles = pad(Angles)
    ADT = split(pad(ADT, 2, jnp.float32), 1)
    DT = pad(DT, 2)
    Trap = pad(Trap, 2)
    Q_bias = split(Q_bias.astype(dtype), 0)
    K_bias = split(K_bias.astype(dtype), 0)

    scale, gamma = [split(x, 1) for x in _compute_scale_gamma(DT, Trap)]
    angle_cumsum = split(angle_dt_cumsum(Angles, DT), 2)
    D = jnp.zeros_like(Q_bias[..., 0]) if D is None else split(D.astype(dtype), 0)

    # vmap over heads, then batch
    fn = jax.vmap(_mamba3_fn, in_axes=(1, 1, 1, 0, 1, 0, 0, 0, 0, 0, None), out_axes=1)
    fn = jax.vmap(fn, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None))
    out = fn(Q, K, V, ADT, angle_cumsum, scale, gamma, D, Q_bias, K_bias, chunk_size)
    out = rearrange(out, "b s h r d -> b s (h r) d")[:, :seqlen]
    return out if Z is None else out * jax.nn.silu(Z.astype(dtype))

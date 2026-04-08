import jax
import jax.numpy as jnp
import pytest

from mamba_ssm.ops.jax.mamba3 import (
    Mamba3SISOState,
    mamba3_siso,
    mamba3_siso_reference,
    mamba3_siso_step_jax,
    mamba3_siso_step_pallas,
)


def _randn(key, shape, dtype=jnp.float32, scale=1.0):
    return scale * jax.random.normal(key, shape, dtype)


def _make_inputs(
    *,
    batch=2,
    seqlen=17,
    nheads=4,
    nheads_qk=2,
    headdim_qk=8,
    headdim_v=6,
    angle_dim=4,
    with_d=True,
    with_z=True,
    with_state=True,
):
    keys = jax.random.split(jax.random.PRNGKey(0), 16)
    q = _randn(keys[0], (batch, seqlen, nheads_qk, headdim_qk))
    k = _randn(keys[1], (batch, seqlen, nheads_qk, headdim_qk))
    v = _randn(keys[2], (batch, seqlen, nheads, headdim_v))
    adt = -jnp.exp(_randn(keys[3], (batch, nheads, seqlen), scale=0.3)).astype(jnp.float32)
    dt = jnp.exp(_randn(keys[4], (batch, nheads, seqlen), scale=0.2) - 2.0).astype(jnp.float32)
    trap = _randn(keys[5], (batch, nheads, seqlen), scale=0.5)
    q_bias = _randn(keys[6], (nheads, headdim_qk))
    k_bias = _randn(keys[7], (nheads, headdim_qk))
    angles = _randn(keys[8], (batch, seqlen, nheads, angle_dim), scale=0.25)
    d = _randn(keys[9], (nheads,), scale=0.1) if with_d else None
    z = _randn(keys[10], (batch, seqlen, nheads, headdim_v), scale=0.2) if with_z else None
    if with_state:
        state = Mamba3SISOState(
            _randn(keys[11], (batch, nheads, angle_dim), dtype=jnp.float32, scale=0.1),
            _randn(keys[12], (batch, nheads, headdim_v, headdim_qk), dtype=jnp.float32, scale=0.1),
            _randn(keys[13], (batch, nheads, headdim_qk), scale=0.1),
            _randn(keys[14], (batch, nheads, headdim_v), scale=0.1),
        )
    else:
        state = None
    return q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, state


@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("with_z", [False, True])
def test_mamba3_siso_matches_reference(with_state, with_z):
    inputs = _make_inputs(with_state=with_state, with_z=with_z)
    q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, state = inputs
    ref_out, ref_state = mamba3_siso_reference(
        q, k, v, adt, dt, trap, q_bias, k_bias, angles, d=d, z=z, initial_state=state
    )
    out, final_state = mamba3_siso(
        q,
        k,
        v,
        adt,
        dt,
        trap,
        q_bias,
        k_bias,
        angles,
        chunk_size=8,
        d=d,
        z=z,
        initial_state=state,
    )
    assert jnp.allclose(out, ref_out, atol=2e-5, rtol=2e-5)
    assert jnp.allclose(final_state.angle_state, ref_state.angle_state, atol=2e-5, rtol=2e-5)
    assert jnp.allclose(final_state.ssm_state, ref_state.ssm_state, atol=2e-5, rtol=2e-5)
    assert jnp.allclose(final_state.k_state, ref_state.k_state, atol=2e-5, rtol=2e-5)
    assert jnp.allclose(final_state.v_state, ref_state.v_state, atol=2e-5, rtol=2e-5)


def test_mamba3_siso_step_pallas_matches_jax():
    inputs = _make_inputs(batch=2, seqlen=5, with_state=True, with_d=True, with_z=True)
    q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, state = inputs

    def run_jax(state):
        outs = []
        for t in range(q.shape[1]):
            out, state = mamba3_siso_step_jax(
                q[:, t],
                k[:, t],
                v[:, t],
                adt[:, :, t],
                dt[:, :, t],
                trap[:, :, t],
                q_bias,
                k_bias,
                angles[:, t],
                state,
                d=d,
                z=z[:, t],
            )
            outs.append(out)
        return jnp.stack(outs, axis=1), state

    def run_pallas(state):
        outs = []
        for t in range(q.shape[1]):
            out, state = mamba3_siso_step_pallas(
                q[:, t],
                k[:, t],
                v[:, t],
                adt[:, :, t],
                dt[:, :, t],
                trap[:, :, t],
                q_bias,
                k_bias,
                angles[:, t],
                state,
                d=d,
                z=z[:, t],
                interpret=True,
            )
            outs.append(out)
        return jnp.stack(outs, axis=1), state

    ref_out, ref_state = run_jax(state)
    out, final_state = run_pallas(state)
    assert jnp.allclose(out, ref_out, atol=2e-5, rtol=2e-5)
    assert jnp.allclose(final_state.angle_state, ref_state.angle_state, atol=2e-5, rtol=2e-5)
    assert jnp.allclose(final_state.ssm_state, ref_state.ssm_state, atol=2e-5, rtol=2e-5)
    assert jnp.allclose(final_state.k_state, ref_state.k_state, atol=2e-5, rtol=2e-5)
    assert jnp.allclose(final_state.v_state, ref_state.v_state, atol=2e-5, rtol=2e-5)

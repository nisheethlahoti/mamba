"""Benchmark: JAX vs Triton Mamba-3 SISO.

Compares correctness, memory usage, and timing.
Run on a machine with an NVIDIA GPU:
    python benchmark_mamba3.py
"""

import time
import gc
from functools import partial

import numpy as np
import torch
import jax
import jax.numpy as jnp

from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined as torch_mamba3
from mamba_ssm.ops.jax.mamba3_siso import mamba3_siso_combined as jax_mamba3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_inputs(batch, seqlen, nheads, nheads_qk, headdim_qk, headdim_v, d_state,
                rope_fraction=0.5, has_D=True, has_Z=True, seed=42):
    """Generate random inputs as numpy arrays."""
    rng = np.random.RandomState(seed)
    headdim_angles = int(d_state * rope_fraction) // 2

    Q = rng.randn(batch, seqlen, nheads_qk, headdim_qk).astype(np.float32) * 0.1
    K = rng.randn(batch, seqlen, nheads_qk, headdim_qk).astype(np.float32) * 0.1
    V = rng.randn(batch, seqlen, nheads, headdim_v).astype(np.float32) * 0.1

    dd_A = rng.randn(batch, seqlen, nheads).astype(np.float32)
    dd_dt = rng.randn(batch, seqlen, nheads).astype(np.float32) - 2.0
    dt_bias = np.abs(rng.randn(nheads).astype(np.float32)) * 0.01

    _A = -np.clip(np.log(1 + np.exp(dd_A)), 1e-4, None)
    DT = np.log(1 + np.exp(dd_dt + dt_bias[None, None, :]))
    ADT = _A * DT
    DT = DT.transpose(0, 2, 1)
    ADT = ADT.transpose(0, 2, 1)

    Trap = rng.randn(batch, nheads, seqlen).astype(np.float32)
    Q_bias = rng.randn(nheads, headdim_qk).astype(np.float32) * 0.1
    K_bias = rng.randn(nheads, headdim_qk).astype(np.float32) * 0.1
    Angles = rng.randn(batch, seqlen, nheads, headdim_angles).astype(np.float32) * 0.1

    D = rng.randn(nheads).astype(np.float32) if has_D else None
    Z = rng.randn(batch, seqlen, nheads, headdim_v).astype(np.float32) * 0.1 if has_Z else None

    return dict(Q=Q, K=K, V=V, ADT=ADT, DT=DT, Trap=Trap,
                Q_bias=Q_bias, K_bias=K_bias, Angles=Angles, D=D, Z=Z)


def to_torch(inputs, device="cuda"):
    return {k: (torch.tensor(v, device=device) if v is not None else None)
            for k, v in inputs.items()}


def to_jax(inputs):
    return {k: (jnp.array(v) if v is not None else None) for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def test_correctness(batch=2, seqlen=2048, nheads=32, nheads_qk=1,
                     headdim_qk=128, headdim_v=64, d_state=128, chunk_size=64):
    print(f"\n{'='*60}")
    print(f"CORRECTNESS TEST")
    print(f"  batch={batch}, seqlen={seqlen}, nheads={nheads}, nheads_qk={nheads_qk}")
    print(f"  headdim_qk={headdim_qk}, headdim_v={headdim_v}, chunk_size={chunk_size}")
    print(f"{'='*60}")

    jit_fn = jax.jit(partial(jax_mamba3, chunk_size=chunk_size))

    for has_D, has_Z, label in [
        (True, True, "D+Z"),
        (True, False, "D only"),
        (False, False, "neither"),
    ]:
        np_inputs = make_inputs(batch, seqlen, nheads, nheads_qk, headdim_qk,
                                headdim_v, d_state, has_D=has_D, has_Z=has_Z)
        t_in = to_torch(np_inputs)
        j_in = to_jax(np_inputs)

        with torch.no_grad():
            t_out = torch_mamba3(
                Q=t_in["Q"], K=t_in["K"], V=t_in["V"],
                ADT=t_in["ADT"], DT=t_in["DT"], Trap=t_in["Trap"],
                Q_bias=t_in["Q_bias"], K_bias=t_in["K_bias"],
                Angles=t_in["Angles"], D=t_in["D"], Z=t_in["Z"],
                chunk_size=chunk_size,
            )
        t_out_np = t_out.float().cpu().numpy()

        j_out = jit_fn(**j_in)
        j_out_np = np.array(j_out)

        max_err = np.max(np.abs(t_out_np - j_out_np))
        mean_err = np.mean(np.abs(t_out_np - j_out_np))
        rel_err = mean_err / (np.mean(np.abs(t_out_np)) + 1e-8)
        ok = max_err < 0.05
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label:10s}  max_err={max_err:.4e}  mean_err={mean_err:.4e}  rel_err={rel_err:.4e}")


# ---------------------------------------------------------------------------
# Memory profiling
# ---------------------------------------------------------------------------

def profile_memory(batch=2, seqlen=2048, nheads=32, nheads_qk=1,
                   headdim_qk=128, headdim_v=64, d_state=128, chunk_size=64):
    """Profile peak GPU memory for JAX forward and forward+backward."""
    print(f"\n{'='*60}")
    print(f"MEMORY PROFILE (JAX)")
    print(f"  batch={batch}, seqlen={seqlen}, nheads={nheads}")
    print(f"{'='*60}")

    device = jax.devices()[0]
    if device.platform != "gpu":
        print("  Skipping memory profiling (no GPU)")
        return

    np_inputs = make_inputs(batch, seqlen, nheads, nheads_qk, headdim_qk,
                            headdim_v, d_state)
    j_in = to_jax(np_inputs)

    # Theoretical minimum
    bf16, f32 = 2, 4
    input_bytes = (
        batch * seqlen * nheads_qk * headdim_qk * bf16 * 2  # Q, K
        + batch * seqlen * nheads * headdim_v * bf16  # V
        + batch * nheads * seqlen * f32 * 3  # ADT, DT, Trap
        + nheads * headdim_qk * f32 * 2  # Q_bias, K_bias
        + batch * seqlen * nheads * (d_state // 4) * bf16  # Angles
        + nheads * f32  # D
        + batch * seqlen * nheads * headdim_v * bf16  # Z
    )
    output_bytes = batch * seqlen * nheads * headdim_v * bf16
    state_bytes = batch * nheads * headdim_v * headdim_qk * f32
    theoretical_min = input_bytes + output_bytes + state_bytes
    print(f"  Theoretical minimum (inputs+output+state): {theoretical_min / 1e6:.1f} MB")

    # --- Forward ---
    jit_fwd = jax.jit(partial(jax_mamba3, chunk_size=chunk_size))
    out = jit_fwd(**j_in)
    out.block_until_ready()

    gc.collect()
    try:
        stats = device.memory_stats()
        peak_fwd = stats["peak_bytes_in_use"]
        print(f"  Forward peak memory: {peak_fwd / 1e6:.1f} MB")
        print(f"  Overhead vs theoretical: {peak_fwd / theoretical_min:.1f}x")
    except Exception as e:
        print(f"  memory_stats() unavailable: {e}")

    # --- Forward + Backward ---
    def _loss(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z):
        return jnp.sum(jax_mamba3(
            Q=Q, K=K, V=V, ADT=ADT, DT=DT, Trap=Trap,
            Q_bias=Q_bias, K_bias=K_bias, Angles=Angles, D=D, Z=Z,
            chunk_size=chunk_size))

    jit_grad = jax.jit(jax.grad(_loss, argnums=(0, 1, 2)))
    grads = jit_grad(
        j_in["Q"], j_in["K"], j_in["V"],
        j_in["ADT"], j_in["DT"], j_in["Trap"],
        j_in["Q_bias"], j_in["K_bias"], j_in["Angles"],
        j_in["D"], j_in["Z"],
    )
    jax.tree.map(lambda x: x.block_until_ready(), grads)

    gc.collect()
    try:
        stats = device.memory_stats()
        peak_bwd = stats["peak_bytes_in_use"]
        print(f"  Forward+Backward peak memory: {peak_bwd / 1e6:.1f} MB")
        print(f"  Overhead vs theoretical: {peak_bwd / theoretical_min:.1f}x")
    except Exception as e:
        print(f"  memory_stats() unavailable: {e}")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def sync_and_wait(out):
    """Synchronize after a forward/backward call."""
    if isinstance(out, jnp.ndarray):
        out.block_until_ready()
    elif isinstance(out, torch.Tensor):
        torch.cuda.synchronize()
    elif isinstance(out, tuple):
        if isinstance(out[0], jnp.ndarray):
            jax.tree.map(lambda x: x.block_until_ready(), out)
        else:
            torch.cuda.synchronize()


def time_fn(fn, n_warmup=5, n_runs=100):
    """Time a function, return median ms."""
    for _ in range(n_warmup):
        sync_and_wait(fn())

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        out = fn()
        sync_and_wait(out)
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return times[len(times) // 2]


def benchmark_timing(batch=2, nheads=32, nheads_qk=1,
                     headdim_qk=128, headdim_v=64, d_state=128, chunk_size=64):
    print(f"\n{'='*60}")
    print(f"TIMING COMPARISON")
    print(f"  batch={batch}, nheads={nheads}, nheads_qk={nheads_qk}")
    print(f"  headdim_qk={headdim_qk}, headdim_v={headdim_v}, chunk_size={chunk_size}")
    print(f"{'='*60}")
    print(f"  {'seqlen':>8s}  {'Triton fwd':>12s}  {'JAX fwd':>12s}  "
          f"{'Triton fwd+bwd':>14s}  {'JAX fwd+bwd':>14s}")
    print(f"  {'':>8s}  {'(ms)':>12s}  {'(ms)':>12s}  {'(ms)':>14s}  {'(ms)':>14s}")
    print(f"  {'-'*70}")

    for seqlen in [512, 1024, 2048, 4096]:
        np_inputs = make_inputs(batch, seqlen, nheads, nheads_qk, headdim_qk,
                                headdim_v, d_state)

        # --- Torch ---
        t_in = to_torch(np_inputs)
        t_in_grad = {k: (v.requires_grad_(True) if v is not None and v.is_floating_point() else v)
                     for k, v in t_in.items()}

        def torch_fwd():
            return torch_mamba3(
                Q=t_in["Q"], K=t_in["K"], V=t_in["V"],
                ADT=t_in["ADT"], DT=t_in["DT"], Trap=t_in["Trap"],
                Q_bias=t_in["Q_bias"], K_bias=t_in["K_bias"],
                Angles=t_in["Angles"], D=t_in["D"], Z=t_in["Z"],
                chunk_size=chunk_size,
            )

        def torch_fwd_bwd():
            for p in t_in_grad.values():
                if p is not None and hasattr(p, 'grad') and p.grad is not None:
                    p.grad = None
            out = torch_mamba3(
                Q=t_in_grad["Q"], K=t_in_grad["K"], V=t_in_grad["V"],
                ADT=t_in_grad["ADT"], DT=t_in_grad["DT"], Trap=t_in_grad["Trap"],
                Q_bias=t_in_grad["Q_bias"], K_bias=t_in_grad["K_bias"],
                Angles=t_in_grad["Angles"], D=t_in_grad["D"], Z=t_in_grad["Z"],
                chunk_size=chunk_size,
            )
            out.sum().backward()
            return out

        # --- JAX ---
        j_in = to_jax(np_inputs)
        jit_fwd = jax.jit(partial(jax_mamba3, chunk_size=chunk_size))

        def jax_fwd():
            return jit_fwd(**j_in)

        def _jax_loss(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z):
            return jnp.sum(jax_mamba3(
                Q=Q, K=K, V=V, ADT=ADT, DT=DT, Trap=Trap,
                Q_bias=Q_bias, K_bias=K_bias, Angles=Angles, D=D, Z=Z,
                chunk_size=chunk_size))

        jit_grad = jax.jit(jax.grad(_jax_loss, argnums=(0, 1, 2)))

        def jax_fwd_bwd():
            return jit_grad(
                j_in["Q"], j_in["K"], j_in["V"],
                j_in["ADT"], j_in["DT"], j_in["Trap"],
                j_in["Q_bias"], j_in["K_bias"], j_in["Angles"],
                j_in["D"], j_in["Z"],
            )

        t_fwd = time_fn(torch_fwd)
        j_fwd = time_fn(jax_fwd)
        t_fb = time_fn(torch_fwd_bwd, n_runs=50)
        j_fb = time_fn(jax_fwd_bwd, n_runs=50)

        print(f"  {seqlen:>8d}  {t_fwd:>12.3f}  {j_fwd:>12.3f}  {t_fb:>14.3f}  {j_fb:>14.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Mamba-3 SISO: JAX vs Triton Benchmark")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    test_correctness()
    profile_memory()
    benchmark_timing()

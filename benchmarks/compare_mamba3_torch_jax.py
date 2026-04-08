from __future__ import annotations

import argparse
import json
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

from mamba_ssm.ops.jax.mamba3 import (
    Mamba3SISOState,
    mamba3_siso,
)
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined


@dataclass
class BenchmarkResult:
    name: str
    ms: float
    compile_ms: float | None = None
    memory_analysis: dict[str, Any] | None = None
    runtime_memory_stats: dict[str, Any] | None = None
    memory_profile: str | None = None
    error: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare torch Triton and JAX Mamba3 SISO.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--nheads", type=int, default=32)
    parser.add_argument("--nheads-qk", type=int, default=4)
    parser.add_argument("--headdim-qk", type=int, default=128)
    parser.add_argument("--headdim-v", type=int, default=64)
    parser.add_argument("--angle-dim", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-dir", type=str, default=None)
    return parser.parse_args()


def _dtype_pair(name: str) -> tuple[torch.dtype, Any]:
    if name == "float16":
        return torch.float16, jnp.float16
    if name == "bfloat16":
        return torch.bfloat16, jnp.bfloat16
    if name == "float32":
        return torch.float32, jnp.float32
    raise ValueError(name)


def _randn(rng: np.random.Generator, shape: tuple[int, ...], scale: float = 1.0) -> np.ndarray:
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _make_numpy_inputs(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    q = _randn(rng, (args.batch, args.seqlen, args.nheads_qk, args.headdim_qk))
    k = _randn(rng, (args.batch, args.seqlen, args.nheads_qk, args.headdim_qk))
    v = _randn(rng, (args.batch, args.seqlen, args.nheads, args.headdim_v))
    adt = -np.exp(_randn(rng, (args.batch, args.nheads, args.seqlen), scale=0.3)).astype(np.float32)
    dt = np.exp(_randn(rng, (args.batch, args.nheads, args.seqlen), scale=0.2) - 2.0).astype(np.float32)
    trap = _randn(rng, (args.batch, args.nheads, args.seqlen), scale=0.5)
    q_bias = _randn(rng, (args.nheads, args.headdim_qk))
    k_bias = _randn(rng, (args.nheads, args.headdim_qk))
    angles = _randn(rng, (args.batch, args.seqlen, args.nheads, args.angle_dim), scale=0.25)
    d = _randn(rng, (args.nheads,), scale=0.1)
    z = _randn(rng, (args.batch, args.seqlen, args.nheads, args.headdim_v), scale=0.2)
    angle_state = _randn(rng, (args.batch, args.nheads, args.angle_dim), scale=0.1)
    ssm_state = _randn(rng, (args.batch, args.nheads, args.headdim_v, args.headdim_qk), scale=0.1)
    k_state = _randn(rng, (args.batch, args.nheads, args.headdim_qk), scale=0.1)
    v_state = _randn(rng, (args.batch, args.nheads, args.headdim_v), scale=0.1)
    return {
        "q": q,
        "k": k,
        "v": v,
        "adt": adt,
        "dt": dt,
        "trap": trap,
        "q_bias": q_bias,
        "k_bias": k_bias,
        "angles": angles,
        "d": d,
        "z": z,
        "angle_state": angle_state,
        "ssm_state": ssm_state,
        "k_state": k_state,
        "v_state": v_state,
    }


def _to_torch(arr: np.ndarray, dtype: torch.dtype, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(arr).to(device=device)
    if tensor.dtype.is_floating_point:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _make_torch_inputs(args: argparse.Namespace, arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
    torch_dtype, _ = _dtype_pair(args.dtype)
    device = "cuda"
    return {
        "Q": _to_torch(arrays["q"], torch_dtype, device),
        "K": _to_torch(arrays["k"], torch_dtype, device),
        "V": _to_torch(arrays["v"], torch_dtype, device),
        "ADT": _to_torch(arrays["adt"], torch.float32, device),
        "DT": _to_torch(arrays["dt"], torch.float32, device),
        "Trap": _to_torch(arrays["trap"], torch_dtype, device),
        "Q_bias": _to_torch(arrays["q_bias"], torch_dtype, device),
        "K_bias": _to_torch(arrays["k_bias"], torch_dtype, device),
        "Angles": _to_torch(arrays["angles"], torch.float32, device),
        "D": _to_torch(arrays["d"], torch.float32, device),
        "Z": _to_torch(arrays["z"], torch_dtype, device),
        "Input_States": (
            _to_torch(arrays["angle_state"], torch.float32, device),
            _to_torch(arrays["ssm_state"], torch.float32, device),
            _to_torch(arrays["k_state"], torch.float32, device),
            _to_torch(arrays["v_state"], torch_dtype, device),
        ),
    }


def _make_jax_inputs(args: argparse.Namespace, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    _, jax_dtype = _dtype_pair(args.dtype)
    return {
        "q": jnp.asarray(arrays["q"], dtype=jax_dtype),
        "k": jnp.asarray(arrays["k"], dtype=jax_dtype),
        "v": jnp.asarray(arrays["v"], dtype=jax_dtype),
        "adt": jnp.asarray(arrays["adt"], dtype=jnp.float32),
        "dt": jnp.asarray(arrays["dt"], dtype=jnp.float32),
        "trap": jnp.asarray(arrays["trap"], dtype=jax_dtype),
        "q_bias": jnp.asarray(arrays["q_bias"], dtype=jax_dtype),
        "k_bias": jnp.asarray(arrays["k_bias"], dtype=jax_dtype),
        "angles": jnp.asarray(arrays["angles"], dtype=jnp.float32),
        "d": jnp.asarray(arrays["d"], dtype=jnp.float32),
        "z": jnp.asarray(arrays["z"], dtype=jax_dtype),
        "initial_state": Mamba3SISOState(
            jnp.asarray(arrays["angle_state"], dtype=jnp.float32),
            jnp.asarray(arrays["ssm_state"], dtype=jnp.float32),
            jnp.asarray(arrays["k_state"], dtype=jax_dtype),
            jnp.asarray(arrays["v_state"], dtype=jax_dtype),
        ),
    }


def _sync_jax(x: Any) -> Any:
    jax.tree_util.tree_map(lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y, x)
    return x


def _runtime_memory_stats(profile_dir: Path | None, tag: str) -> tuple[dict[str, Any] | None, str | None]:
    device = jax.devices()[0]
    stats = device.memory_stats() if hasattr(device, "memory_stats") else None
    profile_path = None
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = str(profile_dir / f"{tag}_memory.prof")
        jax.profiler.save_device_memory_profile(profile_path)
    return stats, profile_path


def _benchmark_torch(name: str, fn, warmup: int, iters: int) -> BenchmarkResult:
    print(f"[torch] {name}: warmup={warmup}, iters={iters}", flush=True)
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000.0 / iters
    del out
    return BenchmarkResult(name=name, ms=ms)


def _benchmark_jax(
    name: str,
    fn,
    args: tuple[Any, ...],
    warmup: int,
    iters: int,
    profile_dir: Path | None,
) -> tuple[Any, BenchmarkResult]:
    print(f"[jax] {name}: lowering", flush=True)
    lowered = jax.jit(fn).lower(*args)
    print(f"[jax] {name}: compiling", flush=True)
    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[jax] {name}: warmup={warmup}", flush=True)
    for _ in range(warmup):
        out = compiled(*args)
        _sync_jax(out)
    print(f"[jax] {name}: timing iters={iters}", flush=True)
    start = time.perf_counter()
    for _ in range(iters):
        out = compiled(*args)
        _sync_jax(out)
    ms = (time.perf_counter() - start) * 1000.0 / iters
    mem = compiled.memory_analysis()
    runtime_stats, profile_path = _runtime_memory_stats(profile_dir, name)
    result = BenchmarkResult(
        name=name,
        ms=ms,
        compile_ms=compile_ms,
        memory_analysis={
            "generated_code_size_in_bytes": mem.generated_code_size_in_bytes,
            "argument_size_in_bytes": mem.argument_size_in_bytes,
            "output_size_in_bytes": mem.output_size_in_bytes,
            "alias_size_in_bytes": mem.alias_size_in_bytes,
            "temp_size_in_bytes": mem.temp_size_in_bytes,
            "host_generated_code_size_in_bytes": mem.host_generated_code_size_in_bytes,
            "host_argument_size_in_bytes": mem.host_argument_size_in_bytes,
            "host_output_size_in_bytes": mem.host_output_size_in_bytes,
            "host_alias_size_in_bytes": mem.host_alias_size_in_bytes,
            "host_temp_size_in_bytes": mem.host_temp_size_in_bytes,
        },
        runtime_memory_stats=runtime_stats,
        memory_profile=profile_path,
    )
    return out, result


def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _torch_forward(inputs: dict[str, Any], chunk_size: int):
    return mamba3_siso_combined(
        Q=inputs["Q"],
        K=inputs["K"],
        V=inputs["V"],
        ADT=inputs["ADT"],
        DT=inputs["DT"],
        Trap=inputs["Trap"],
        Q_bias=inputs["Q_bias"],
        K_bias=inputs["K_bias"],
        Angles=inputs["Angles"],
        D=inputs["D"],
        Z=inputs["Z"],
        Input_States=inputs["Input_States"],
        chunk_size=chunk_size,
        return_final_states=True,
    )


def _jax_forward(inputs: dict[str, Any], chunk_size: int):
    return mamba3_siso(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["adt"],
        inputs["dt"],
        inputs["trap"],
        inputs["q_bias"],
        inputs["k_bias"],
        inputs["angles"],
        chunk_size=chunk_size,
        d=inputs["d"],
        z=inputs["z"],
        initial_state=inputs["initial_state"],
    )


def _print_json(title: str, payload: Any) -> None:
    print(title)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _format_exception(exc: Exception) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is required for the Triton comparison.")
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"JAX backend is {jax.default_backend()!r}; run this script on a machine where JAX sees the NVIDIA GPU."
        )
    if args.headdim_qk % 2 != 0:
        raise ValueError("headdim-qk must be even for rotary.")
    if args.angle_dim > args.headdim_qk // 2:
        raise ValueError("angle-dim must be <= headdim-qk // 2.")
    if args.nheads % args.nheads_qk != 0:
        raise ValueError("nheads must be divisible by nheads-qk.")

    profile_dir = Path(args.profile_dir) if args.profile_dir else Path(tempfile.mkdtemp(prefix="mamba3_jax_profile_"))
    arrays = _make_numpy_inputs(args)
    print(
        f"Config: batch={args.batch} seqlen={args.seqlen} nheads={args.nheads} "
        f"nheads_qk={args.nheads_qk} headdim_qk={args.headdim_qk} "
        f"headdim_v={args.headdim_v} chunk_size={args.chunk_size} dtype={args.dtype}",
        flush=True,
    )
    torch_inputs = _make_torch_inputs(args, arrays)
    jax_inputs = _make_jax_inputs(args, arrays)

    correctness: dict[str, Any] = {}
    timings: list[BenchmarkResult] = []

    print("[check] forward correctness", flush=True)
    torch_out, torch_angle, torch_ssm, torch_k, torch_v = _torch_forward(torch_inputs, args.chunk_size)
    jax_out, jax_state = _jax_forward(jax_inputs, args.chunk_size)
    _sync_jax((jax_out, jax_state))
    correctness["forward"] = {
        "out_max_abs_diff": _max_diff(np.asarray(torch_out.detach().float().cpu()), np.asarray(jax_out)),
        "angle_state_max_abs_diff": _max_diff(np.asarray(torch_angle.detach().float().cpu()), np.asarray(jax_state.angle_state)),
        "ssm_state_max_abs_diff": _max_diff(np.asarray(torch_ssm.detach().float().cpu()), np.asarray(jax_state.ssm_state)),
        "k_state_max_abs_diff": _max_diff(np.asarray(torch_k.detach().float().cpu()), np.asarray(jax_state.k_state)),
        "v_state_max_abs_diff": _max_diff(np.asarray(torch_v.detach().float().cpu()), np.asarray(jax_state.v_state)),
    }

    timings.append(_benchmark_torch("torch_triton_forward", lambda: _torch_forward(torch_inputs, args.chunk_size), args.warmup, args.iters))
    _, jax_result = _benchmark_jax(
        "jax_forward",
        lambda *fn_args: _jax_forward(
            {
                "q": fn_args[0],
                "k": fn_args[1],
                "v": fn_args[2],
                "adt": fn_args[3],
                "dt": fn_args[4],
                "trap": fn_args[5],
                "q_bias": fn_args[6],
                "k_bias": fn_args[7],
                "angles": fn_args[8],
                "d": fn_args[9],
                "z": fn_args[10],
                "initial_state": fn_args[11],
            },
            args.chunk_size,
        ),
        (
            jax_inputs["q"],
            jax_inputs["k"],
            jax_inputs["v"],
            jax_inputs["adt"],
            jax_inputs["dt"],
            jax_inputs["trap"],
            jax_inputs["q_bias"],
            jax_inputs["k_bias"],
            jax_inputs["angles"],
            jax_inputs["d"],
            jax_inputs["z"],
            jax_inputs["initial_state"],
        ),
        args.warmup,
        args.iters,
        profile_dir,
    )
    timings.append(jax_result)

    summary = {
        "config": vars(args),
        "jax_backend": jax.default_backend(),
        "torch_device": str(torch.device("cuda")),
        "correctness": correctness,
        "timings": [asdict(x) for x in timings],
    }
    _print_json("Mamba3 Torch vs JAX", summary)
    print(f"JAX memory profiles written under: {profile_dir}")


if __name__ == "__main__":
    main()

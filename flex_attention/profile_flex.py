import logging
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Callable

import torch

from torch.nn.attention.flex_attention import flex_attention
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("flex_bench")
logger.setLevel(logging.INFO)
Tensor = torch.Tensor


# source: https://gist.github.com/drisspg/4f125f199ae670aaeeb26bb0cde12a79
def per_tensor_scaling(score, b, h, q_idx, k_idx, q_scale, k_scale):
    return q_scale * k_scale * score.to(torch.torch.float32)


# source: https://gist.github.com/drisspg/4f125f199ae670aaeeb26bb0cde12a79
def scaled_flex(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    scale_func: Callable,
    **kwargs,
):
    scale_func_partial = partial(scale_func, q_scale=q_scale, k_scale=k_scale)
    out = flex_attention(query, key, value, score_mod=scale_func_partial, **kwargs)
    out = out.to(torch.bfloat16)
    out *= v_scale
    return out


# scaled flex attention emulates quantizing the to a lower precision dtype
compiled_scale_flex = torch.compile(scaled_flex, fullgraph=True, dynamic=False)
compiled_flex = torch.compile(flex_attention, fullgraph=True, dynamic=False)

torch._dynamo.config.cache_size_limit = 1000


def profile_flex(test_name, compiled_flex, dtype, trace=True, **kwargs):
    B, H, seq_q, seq_k, D = 16, 16, 1024, 1024, 128

    # construct q/k/v
    make_tensor = partial(torch.randn, device="cuda", dtype=torch.bfloat16)
    query = make_tensor((B, H, seq_q, D)).to(dtype)
    key = make_tensor((B, H, seq_k, D)).to(dtype)
    value = make_tensor((B, H, seq_k, D)).to(dtype)

    # generate trace
    if trace:
        logger.info(f"Generating trace...")
        wait, warmup, active = 1, 10, 1
        total_steps = wait + warmup + active
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=0
            ),
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(total_steps):
                out = compiled_flex(query, key, value, **kwargs)
                prof.step()

        # save trace
        _ = prof.export_chrome_trace(f"{test_name}.json")
        logger.info(f"Trace saved to {test_name}.json")

    # run benchmark (already warmed up from profiling warmup steps)
    logger.info(f"Running benchmark: {test_name}")
    bench_time = benchmark_cuda_function_in_microseconds(
        compiled_flex, query, key, value, **kwargs
    )
    logger.info(f"{test_name}: {bench_time} us")


def main(args: Namespace):
    # scales for fp8
    q_scale = torch.tensor(1, device="cuda", dtype=torch.bfloat16)  # float32
    k_scale = torch.tensor(1, device="cuda", dtype=torch.bfloat16)  # float32
    v_scale = torch.tensor(1, device="cuda", dtype=torch.bfloat16)  # float32

    test_cases = []
    if args.bf16:
        test_cases.append(
            {
                "test_name": "bf16",
                "dtype": torch.bfloat16,
                "func": compiled_flex,
                "kwargs": {},
            }
        )
    if args.fp8:
        test_cases.append(
            {
                "test_name": "fp8e4m3",
                "dtype": torch.float8_e4m3fn,
                "func": compiled_scale_flex,
                "kwargs": {
                    "q_scale": q_scale,
                    "k_scale": k_scale,
                    "v_scale": v_scale,
                    "scale_func": per_tensor_scaling,
                },
            }
        )

    for test_case in test_cases:
        test_name, dtype, func, kwargs = (
            test_case["test_name"],
            test_case["dtype"],
            test_case["func"],
            test_case["kwargs"],
        )
        profile_flex(test_name, func, dtype, trace=args.trace, **kwargs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    args = parser.parse_args()
    main(args)

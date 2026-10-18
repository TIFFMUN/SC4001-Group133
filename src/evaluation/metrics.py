"""
metrics.py
----------
Shared metric utilities used by eval_gsm8k.py and eval_arc.py.

Metrics tracked across all experiments:
    - Accuracy     : % of correct final answers
    - Latency      : wall-clock time per sample (ms)
    - Token count  : output tokens per sample (proxy for compute cost)

Member responsible: C
"""

import time
import logging
from typing import Callable

import torch
import pandas as pd

logger = logging.getLogger(__name__)


def compute_latency(
    model,
    inputs: dict,
    max_new_tokens: int,
    do_sample: bool = False,
) -> tuple[float, torch.Tensor]:
    """
    Run model.generate() and return (latency_ms, output_ids).

    Latency is wall-clock time in milliseconds for the full generation call.
    This is the primary computational cost metric for the report.

    TODO: for a more rigorous compute cost estimate, also record:
          - number of output tokens (already captured separately)
          - GPU memory usage via torch.cuda.max_memory_allocated()
    """
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=model.config.eos_token_id,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, output_ids


def compute_accuracy(results: list[dict], correct_key: str = "is_correct") -> float:
    """Compute accuracy from a list of result dicts."""
    if not results:
        return 0.0
    n_correct = sum(1 for r in results if r[correct_key])
    return n_correct / len(results)


def compute_summary_stats(results: list[dict]) -> dict:
    """
    Compute summary statistics across all results for the report table.

    Returns a dict with:
        accuracy, mean_latency_ms, median_latency_ms, mean_token_count
    """
    accuracy = compute_accuracy(results)
    latencies = [r["latency_ms"] for r in results]
    tokens = [r["token_count"] for r in results]

    return {
        "accuracy": round(accuracy, 4),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
        "median_latency_ms": round(sorted(latencies)[len(latencies) // 2], 2),
        "mean_token_count": round(sum(tokens) / len(tokens), 2),
        "n_samples": len(results),
    }


def save_results(results: list[dict], output_path: str) -> None:
    """
    Save raw per-sample results to a CSV file.
    Use this for every experiment run so results can be compared in notebook 04.

    TODO: name output files descriptively, e.g.:
          exp0_zero_shot_gsm8k.csv
          exp2_adaptation_r16_gsm8k.csv
          exp5_filtered_distillation_arc.csv
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path} ({len(df)} rows)")


def print_summary(experiment_name: str, stats: dict) -> None:
    """Pretty-print summary stats for a completed experiment."""
    logger.info(
        f"\n{'='*50}\n"
        f"Experiment: {experiment_name}\n"
        f"  Accuracy:          {stats['accuracy']:.2%}\n"
        f"  Mean latency:      {stats['mean_latency_ms']:.1f} ms/sample\n"
        f"  Median latency:    {stats['median_latency_ms']:.1f} ms/sample\n"
        f"  Mean token count:  {stats['mean_token_count']:.1f} tokens/sample\n"
        f"  Samples:           {stats['n_samples']}\n"
        f"{'='*50}"
    )

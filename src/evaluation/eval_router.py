"""
eval_router.py
--------------
Exp 6 (R1): Inference-gated budget router evaluation.

This is a pure inference-time experiment — no additional training.
Prerequisite: at least one trained checkpoint from Exps 2–4 (Exp 4 recommended).

Workflow:
  1. Load the Exp 4 (filtered distillation) checkpoint.
  2. Score each test prompt with compute_complexity() — one forward pass, no generation.
  3. Route to easy profile (low max_new_tokens) or hard profile (high max_new_tokens).
  4. Sweep the routing threshold across N values to build a Pareto curve.
  5. Save per-sample CSVs and a sweep summary CSV for plotting in notebook 04.

Usage:
    python -m src.evaluation.eval_router --experiment exp6_router
    python -m src.evaluation.eval_router --experiment exp6_router --max_samples 200
"""

import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd

from src.distillation.filter_traces import extract_answer, answers_match
from src.evaluation.budget_router import BudgetRouter, compute_complexity
from src.evaluation.metrics import compute_summary_stats, save_results, print_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model_for_routing(cfg: dict, checkpoint_override: str | None = None):
    """
    Load model for routing. Mirrors eval_gsm8k.load_model() but reads from the
    router: section of eval.yaml rather than the top-level model: section.
    """
    base_model_name = cfg["model"]["base_model"]
    router_cfg = cfg.get("router", {})
    checkpoint_path = checkpoint_override or router_cfg.get("checkpoint_path") or cfg["model"].get("checkpoint_path")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading LoRA adapter from {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
    elif checkpoint_path:
        logger.warning(
            f"Checkpoint '{checkpoint_path}' not found on disk. "
            f"Evaluating pre-trained base model '{base_model_name}'."
        )

    model.eval()
    return model, tokenizer


def build_prompt(question: str, tokenizer) -> str:
    """Same zero-shot prompt as eval_gsm8k.py for a fair comparison."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Solve problems step by step. "
                       "At the end, write your final answer as: #### <number>",
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def collect_complexity_scores(
    model,
    tokenizer,
    test_data,
    max_samples: int | None,
) -> np.ndarray:
    """
    Dry run: compute complexity scores for all samples without generating.
    Used to set the threshold grid based on actual score distribution.
    """
    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))

    scores = []
    with torch.no_grad():
        for sample in tqdm(test_data, desc="Scoring prompt complexity"):
            prompt = build_prompt(sample["question"], tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            scores.append(compute_complexity(model, inputs))

    return np.array(scores)


def evaluate_at_threshold(
    router: BudgetRouter,
    test_data,
    experiment_name: str,
    output_dir: Path,
) -> dict:
    """
    Run one evaluation pass with the router's current threshold.
    Returns summary stats for that threshold.
    """
    results = []

    with torch.no_grad():
        for sample in tqdm(test_data, desc=f"Evaluating [threshold={router.threshold:.3f}]", leave=False):
            question = sample["question"]
            ground_truth = sample["answer"]

            prompt = build_prompt(question, tokenizer=router.tokenizer)
            result = router.route(prompt)

            predicted_answer = extract_answer(result.prediction)
            is_correct = answers_match(predicted_answer, ground_truth)

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": result.prediction,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "latency_ms": result.latency_ms,
                "token_count": result.token_count,
                "route": result.route,
                "complexity": result.complexity,
                "threshold": router.threshold,
            })

    stats = compute_summary_stats(results)
    n_easy = sum(1 for r in results if r["route"] == "easy")
    n_hard = len(results) - n_easy

    return {
        "threshold": router.threshold,
        "accuracy": stats["accuracy"],
        "mean_token_count": stats["mean_token_count"],
        "mean_latency_ms": stats["mean_latency_ms"],
        "n_easy": n_easy,
        "n_hard": n_hard,
        "pct_easy": round(n_easy / len(results), 4),
        "n_samples": stats["n_samples"],
    }, results


def run_router_sweep(
    config_path: str = "configs/eval.yaml",
    experiment_name: str = "exp6_router",
    max_samples: int | None = None,
    checkpoint_override: str | None = None,
) -> pd.DataFrame:
    """
    Main entry point. Sweeps the routing threshold and saves a Pareto-ready CSV.

    Args:
        config_path:         path to eval.yaml (must have router: section)
        experiment_name:     prefix for output CSV names
        max_samples:         override max_samples for quick testing
        checkpoint_override: optional path to trained LoRA checkpoint

    Returns:
        DataFrame with one row per threshold: (threshold, accuracy, mean_token_count, ...)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    router_cfg = cfg.get("router", {})
    if not router_cfg.get("enabled", False) and not checkpoint_override:
        logger.warning(
            "router.enabled is false in eval.yaml. "
            "Set router.enabled: true to run Exp 6."
        )

    output_dir = Path(cfg["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Generation profiles ---
    easy_kwargs = {
        "max_new_tokens": router_cfg.get("easy_max_tokens") or 128,
        "do_sample": False,
    }
    hard_kwargs = {
        "max_new_tokens": router_cfg.get("hard_max_tokens") or 300,
        "do_sample": False,
    }

    # --- Load dataset ---
    logger.info("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"]

    n_samples = max_samples or router_cfg.get("max_samples") or cfg["gsm8k"].get("max_samples")
    if n_samples:
        test_data = test_data.select(range(min(n_samples, len(test_data))))
        logger.info(f"Using {len(test_data)} samples.")

    # --- Load model ---
    model, tokenizer = load_model_for_routing(cfg, checkpoint_override=checkpoint_override)

    # --- Dry-run: collect complexity scores to calibrate threshold grid ---
    logger.info("Computing complexity scores for threshold calibration...")
    scores = collect_complexity_scores(model, tokenizer, test_data, max_samples=None)

    threshold_min = router_cfg.get("threshold_min")
    if threshold_min is None:
        threshold_min = float(np.percentile(scores, 10))

    threshold_max = router_cfg.get("threshold_max")
    if threshold_max is None:
        threshold_max = float(np.percentile(scores, 90))

    n_thresholds = router_cfg.get("threshold_steps") or 10
    thresholds = np.linspace(threshold_min, threshold_max, n_thresholds).tolist()

    # Add anchor points: all-easy (threshold → ∞) and all-hard (threshold → -∞)
    thresholds = [float("-inf")] + thresholds + [float("inf")]

    logger.info(
        f"Score distribution: p10={threshold_min:.3f}, p90={threshold_max:.3f}\n"
        f"Sweeping {len(thresholds)} thresholds."
    )

    # --- Threshold sweep ---
    sweep_rows = []
    for threshold in thresholds:
        router = BudgetRouter(
            model=model,
            tokenizer=tokenizer,
            easy_kwargs=easy_kwargs,
            hard_kwargs=hard_kwargs,
            threshold=threshold,
        )
        sweep_row, per_sample_results = evaluate_at_threshold(
            router, test_data, experiment_name, output_dir
        )
        sweep_rows.append(sweep_row)

        # Save per-sample CSV for the extreme anchors and best-guess mid-threshold
        tag = f"{threshold:.3f}".replace("-inf", "all_hard").replace("inf", "all_easy")
        if threshold in (float("-inf"), float("inf")):
            per_sample_path = output_dir / f"{experiment_name}_t{tag}_gsm8k.csv"
            save_results(per_sample_results, str(per_sample_path))

        logger.info(
            f"threshold={tag:<12} | acc={sweep_row['accuracy']:.3f} "
            f"| mean_tokens={sweep_row['mean_token_count']:.1f} "
            f"| easy={sweep_row['pct_easy']:.1%}"
        )

    # --- Save sweep summary ---
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = output_dir / f"{experiment_name}_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    logger.info(f"Sweep results saved to {sweep_path}")

    # Find Pareto-optimal points (no other point has both lower cost and higher accuracy)
    sweep_df["pareto_optimal"] = False
    for i, row in sweep_df.iterrows():
        dominated = (
            (sweep_df["mean_token_count"] <= row["mean_token_count"]) &
            (sweep_df["accuracy"] >= row["accuracy"]) &
            ~((sweep_df["mean_token_count"] == row["mean_token_count"]) &
              (sweep_df["accuracy"] == row["accuracy"]))
        )
        if not dominated.any():
            sweep_df.at[i, "pareto_optimal"] = True

    sweep_df.to_csv(sweep_path, index=False)

    n_pareto = sweep_df["pareto_optimal"].sum()
    logger.info(f"Pareto-optimal points: {n_pareto} / {len(sweep_df)}")

    return sweep_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="exp6_router",
                        help="Prefix for output CSV filenames.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of test samples (useful for quick sanity checks).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (e.g. results/checkpoints/lora_r16).")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()

    run_router_sweep(
        config_path=args.config,
        experiment_name=args.experiment,
        max_samples=args.max_samples,
        checkpoint_override=args.checkpoint,
    )

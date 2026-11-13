"""
eval_gsm8k.py
-------------
Evaluates a model (base or LoRA-finetuned) on the GSM8K test set.

Used for: Exp 0, 1, 2, 3, 4 (all experiments on the math benchmark).
Exp 5 (cross-task transfer) uses eval_arc.py on the Exp 4 checkpoint.

Answer parsing:
    GSM8K ground truth always contains '#### <number>'.
    We extract the predicted answer using the same pattern from the model output.
    This is the same logic as filter_traces.py — kept consistent on purpose.

Member responsible: C
"""

import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.distillation.filter_traces import extract_answer, answers_match
from src.evaluation.metrics import compute_latency, compute_summary_stats, save_results, print_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model(cfg: dict, checkpoint_override: str | None = None):
    """
    Load base model or LoRA-finetuned model depending on eval.yaml config.

    If checkpoint_path is null → evaluate base model (Exp 0, 1).
    If checkpoint_path is set  → load LoRA adapter on top of base model (Exp 2-5).
    """
    base_model_name = cfg["model"]["base_model"]
    checkpoint_path = checkpoint_override or cfg["model"].get("checkpoint_path")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path if checkpoint_path else base_model_name
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if checkpoint_path:
        logger.info(f"Loading LoRA adapter from {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()  # merge LoRA weights for faster inference

    model.eval()
    return model, tokenizer


def build_prompt(question: str, tokenizer) -> str:
    """
    Zero-shot prompt for evaluation.
    All experiments (Exp 0-5) use the same evaluation prompt for fair comparison.

    TODO: confirm this matches the format used during training in finetune.py
          to avoid train/eval prompt mismatch
    """
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


def evaluate_gsm8k(
    config_path: str = "configs/eval.yaml",
    experiment_name: str = "unnamed",
    max_samples: int | None = None,
    checkpoint_override: str | None = None,
) -> list[dict]:
    """
    Main entry point. Runs evaluation on the full GSM8K test set.

    Args:
        config_path:         path to eval.yaml
        experiment_name:     used for naming the output CSV, e.g. "exp2_adaptation_r16"
        max_samples:         override max_samples for quick testing
        checkpoint_override: optional path to trained checkpoint

    Returns:
        List of per-sample result dicts.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if not cfg["benchmarks"].get("gsm8k", True):
        logger.info("GSM8K evaluation disabled in eval.yaml. Skipping.")
        return []

    gsm8k_cfg = cfg["gsm8k"]
    gen_cfg = cfg["generation"]
    output_dir = Path(cfg["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    logger.info("Loading GSM8K test set...")
    dataset = load_dataset(gsm8k_cfg["dataset"], gsm8k_cfg["split"])

    # GSM8K "test" is a key in the loaded dict
    test_data = dataset["test"]
    n_samples = max_samples or gsm8k_cfg.get("max_samples")
    if n_samples:
        test_data = test_data.select(range(min(n_samples, len(test_data))))
        logger.info(f"Limiting to {len(test_data)} samples.")

    # --- Load model ---
    model, tokenizer = load_model(cfg, checkpoint_override=checkpoint_override)

    # --- Evaluate ---
    results = []

    with torch.no_grad():
        for sample in tqdm(test_data, desc=f"Evaluating GSM8K [{experiment_name}]"):
            question = sample["question"]
            ground_truth = sample["answer"]

            prompt = build_prompt(question, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]

            latency_ms, output_ids = compute_latency(
                model=model,
                inputs=inputs,
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample=gen_cfg["do_sample"],
            )

            generated_ids = output_ids[0][input_len:]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
            token_count = len(generated_ids)

            predicted_answer = extract_answer(prediction)
            is_correct = answers_match(predicted_answer, ground_truth)

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "latency_ms": latency_ms,
                "token_count": token_count,
            })

    # --- Save and summarise ---
    output_path = output_dir / f"{experiment_name}_gsm8k.csv"
    save_results(results, str(output_path))

    stats = compute_summary_stats(results)
    print_summary(experiment_name, stats)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="exp0_zero_shot",
                        help="Descriptive name for this run, used in output filename.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of test samples (useful for quick sanity checks).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (e.g. results/checkpoints/lora_r16).")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()

    evaluate_gsm8k(
        config_path=args.config,
        experiment_name=args.experiment,
        max_samples=args.max_samples,
        checkpoint_override=args.checkpoint,
    )

"""
eval_arc.py
-----------
Evaluates a model on the ARC-Challenge test set.

Primary use: Exp 5 (D3) — cross-task transfer.
The model trained on GSM8K distillation data (Exp 4) is evaluated here
to test whether math reasoning distillation transfers to commonsense reasoning.

ARC-Challenge format:
    Multiple choice, 4 options (A/B/C/D or 1/2/3/4).
    We compare per-token log-probabilities for each answer choice
    (likelihood scoring) rather than generating free text.
    This is more reliable for multiple choice than extracting a letter from generated text.

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

from src.evaluation.metrics import compute_summary_stats, save_results, print_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model(cfg: dict, checkpoint_override: str | None = None):
    """Load base or LoRA-finetuned model. Same logic as eval_gsm8k.py."""
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
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def format_question(sample: dict) -> tuple[str, list[str], str]:
    """
    Extract question text, answer choices, and correct answer key from ARC sample.

    Returns:
        question:  the question text
        choices:   list of answer text strings
        answer_key: the correct letter/number (e.g. "A", "B", "1", "2")
    """
    question = sample["question"]
    choices = sample["choices"]["text"]
    labels = sample["choices"]["label"]
    answer_key = sample["answerKey"]
    return question, choices, labels, answer_key


def score_choice(
    model,
    tokenizer,
    question: str,
    choice_text: str,
) -> float:
    """
    Compute the log-probability of a single answer choice given the question.

    We score each choice as:
        log P(choice | question)
    and pick the choice with the highest score as the model's prediction.

    This likelihood-scoring approach avoids the need to parse a letter from
    free-form generation, which is unreliable for small models.

    TODO: experiment with prompt format — does adding "Answer:" before the
          choice improve discrimination between options?
    """
    prompt = (
        f"Question: {question}\n"
        f"Answer: {choice_text}"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    # outputs.loss is the mean negative log-likelihood over all tokens
    # We return the negative loss (= log-likelihood) so higher = better
    return -outputs.loss.item()


def evaluate_arc(
    config_path: str = "configs/eval.yaml",
    experiment_name: str = "unnamed",
    max_samples: int | None = None,
    checkpoint_override: str | None = None,
) -> list[dict]:
    """
    Main entry point. Runs likelihood-scoring evaluation on ARC-Challenge test set.

    Args:
        config_path:         path to eval.yaml
        experiment_name:     used for naming the output CSV, e.g. "exp5_d3_cross_task"
        max_samples:         override max_samples for quick testing
        checkpoint_override: optional path to trained checkpoint

    Returns:
        List of per-sample result dicts.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if not cfg["benchmarks"].get("arc", True):
        logger.info("ARC evaluation disabled in eval.yaml. Skipping.")
        return []

    arc_cfg = cfg["arc"]
    output_dir = Path(cfg["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    logger.info("Loading ARC-Challenge test set...")
    dataset = load_dataset(arc_cfg["dataset"], arc_cfg["config"])
    test_data = dataset["test"]

    n_samples = max_samples or arc_cfg.get("max_samples")
    if n_samples:
        test_data = test_data.select(range(min(n_samples, len(test_data))))
        logger.info(f"Limiting to {len(test_data)} samples.")

    # --- Load model ---
    model, tokenizer = load_model(cfg, checkpoint_override=checkpoint_override)

    # --- Evaluate ---
    results = []

    for sample in tqdm(test_data, desc=f"Evaluating ARC [{experiment_name}]"):
        question, choices, labels, answer_key = format_question(sample)

        # Score each answer choice
        import time
        t0 = time.perf_counter()
        scores = [
            score_choice(model, tokenizer, question, choice)
            for choice in choices
        ]
        latency_ms = (time.perf_counter() - t0) * 1000

        # Predicted label = choice with highest log-probability
        best_idx = scores.index(max(scores))
        predicted_label = labels[best_idx]
        is_correct = predicted_label == answer_key

        # Token count: sum of choice lengths as a proxy for compute cost
        token_count = sum(
            len(tokenizer(c, return_tensors="pt")["input_ids"][0]) for c in choices
        )

        results.append({
            "question": question,
            "choices": choices,
            "answer_key": answer_key,
            "predicted_label": predicted_label,
            "is_correct": is_correct,
            "scores": scores,
            "latency_ms": latency_ms,
            "token_count": token_count,
        })

    # --- Save and summarise ---
    # Flatten choices/scores for CSV serialisation
    flat_results = [
        {k: (str(v) if isinstance(v, list) else v) for k, v in r.items()}
        for r in results
    ]
    output_path = output_dir / f"{experiment_name}_arc.csv"
    save_results(flat_results, str(output_path))

    stats = compute_summary_stats(results)
    print_summary(experiment_name, stats)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="exp5_d3_cross_task",
                        help="Descriptive name for this run, used in output filename.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of test samples (useful for quick sanity checks).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (e.g. results/checkpoints/lora_r16).")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()

    evaluate_arc(
        config_path=args.config,
        experiment_name=args.experiment,
        max_samples=args.max_samples,
        checkpoint_override=args.checkpoint,
    )

"""
cot_prompting.py
----------------
Experiments: Exp 0 (zero-shot) and Exp 1 (few-shot CoT)

Runs the base Qwen2.5-0.5B-Instruct model with no fine-tuning.
These form the lower-bound baselines that all trained models (Exp 2-5) are compared against.

Exp 0: Zero-shot — just ask the question, no examples.
Exp 1: Few-shot CoT — prepend 4 worked examples to guide the model's reasoning format.

Member responsible: C
"""

import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.evaluation.metrics import compute_latency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Few-shot examples (Exp 1)
# These 4 examples are drawn from the GSM8K training set.
# They demonstrate the #### answer format and step-by-step reasoning style.
# TODO: experiment with different numbers of shots (2, 4, 8) and report if
#       it significantly affects accuracy vs token cost
# =============================================================================
FEW_SHOT_EXAMPLES = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48 + 24 = 72 clips altogether in April and May.\n#### 72",
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 × 50 = $10.\n#### 10",
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": "Betty has 100/2 = $50.\nHer grandparents gave her 15 × 2 = $30.\nBetty has 50 + 15 + 30 = $95.\nShe needs 100 - 95 = $5 more.\n#### 5",
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "answer": "Today Julie read 12 × 2 = 24 pages.\nSo she has read 12 + 24 = 36 pages so far.\nRemaining: 120 - 36 = 84 pages.\nTomorrow she should read 84/2 = 42 pages.\n#### 42",
    },
]


def build_zero_shot_prompt(question: str, tokenizer) -> str:
    """
    Exp 0: Zero-shot prompt — no examples, just the question.
    The model must rely entirely on its pretrained knowledge.
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


def build_few_shot_prompt(question: str, tokenizer) -> str:
    """
    Exp 1: Few-shot CoT prompt — prepend worked examples before the question.
    The examples demonstrate the reasoning format the model should follow.

    TODO: try with 2, 4, and 8 examples and report accuracy vs input token cost
          to show the inference cost side of the accuracy/cost trade-off
    """
    # Build a single user message with all examples + the question
    examples_text = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += f"Example {i}:\nQuestion: {ex['question']}\nSolution: {ex['answer']}\n\n"

    full_user_message = (
        f"{examples_text}"
        f"Now solve this problem step by step:\nQuestion: {question}"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Solve problems step by step. "
                       "At the end, write your final answer as: #### <number>",
        },
        {"role": "user", "content": full_user_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_baseline(
    mode: str = "zero_shot",
    dataset_name: str = "gsm8k",
    max_samples: int | None = None,
    config_path: str = "configs/eval.yaml",
) -> list[dict]:
    """
    Run Exp 0 or Exp 1 on GSM8K or ARC test sets and return raw results.

    Args:
        mode:         "zero_shot" (Exp 0) or "few_shot" (Exp 1)
        dataset_name: "gsm8k" or "arc"
        max_samples:  limit samples for quick testing; None = full test set
        config_path:  path to eval.yaml

    Returns:
        List of dicts with keys: question, prediction, ground_truth, latency_ms, token_count
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["base_model"]
    gen_cfg = cfg["generation"]

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    # --- Load dataset ---
    if dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")["test"]
        question_key, answer_key = "question", "answer"
    elif dataset_name == "arc":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
        question_key, answer_key = "question", "answerKey"
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose: gsm8k, arc.")

    if max_samples:
        dataset = dataset.select(range(max_samples))

    # --- Prompt builder ---
    build_prompt = (
        build_zero_shot_prompt if mode == "zero_shot" else build_few_shot_prompt
    )

    results = []

    with torch.no_grad():
        for sample in tqdm(dataset, desc=f"{mode} on {dataset_name}"):
            question = sample[question_key]
            ground_truth = sample[answer_key]

            prompt = build_prompt(question, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]

            latency_ms, output_ids = compute_latency(
                model=model,
                inputs=inputs,
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample=gen_cfg["do_sample"],
            )

            # Decode only the generated tokens (not the prompt)
            generated_ids = output_ids[0][input_len:]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
            token_count = len(generated_ids)

            results.append({
                "question": question,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "latency_ms": latency_ms,
                "token_count": token_count,
            })

    logger.info(f"Baseline run complete: {len(results)} samples.")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--dataset", choices=["gsm8k", "arc"], default="gsm8k")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    results = run_baseline(
        mode=args.mode,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
    )
    logger.info(f"Sample result: {results[0]}")

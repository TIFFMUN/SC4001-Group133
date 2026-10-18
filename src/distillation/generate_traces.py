"""
generate_traces.py
------------------
Experiment: D1 (standard distillation), D2 (filtered distillation)

Calls the Groq API to generate chain-of-thought (CoT) reasoning traces
from a teacher model (Qwen2.5-7B via Groq) for every question in the
GSM8K training set. Traces are saved as JSONL for downstream filtering
and dataset construction.

Member responsible: A
"""

import os
import time
import json
import logging
from pathlib import Path

import yaml
from groq import Groq
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/distillation.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_prompt(question: str, template: str) -> str:
    """Format the question into the teacher prompt template."""
    return template.format(question=question)


def call_teacher(
    client: Groq,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str | None:
    """
    Send a single prompt to the Groq API and return the response text.
    Returns None on failure so the caller can decide how to handle it.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq API call failed: {e}")
        return None


def generate_traces(config_path: str = "configs/distillation.yaml") -> None:
    """
    Main entry point.
    Loads GSM8K train split, generates a CoT trace per question via the
    Groq teacher API, and writes results to JSONL.

    Each line in the output JSONL has:
        {
            "question": str,
            "ground_truth": str,   # the #### answer from GSM8K
            "trace": str           # teacher-generated CoT + answer
        }
    """
    cfg = load_config(config_path)
    teacher_cfg = cfg["teacher"]
    dataset_cfg = cfg["dataset"]
    output_cfg = cfg["output"]
    prompt_template = cfg["prompt_template"]

    # --- Load dataset ---
    logger.info("Loading GSM8K training set...")
    dataset = load_dataset(dataset_cfg["name"], dataset_cfg["split"])
    train_data = dataset[dataset_cfg["train_split"]]

    # Optionally limit samples for quick testing
    # TODO: set max_samples in distillation.yaml to ~100 for a quick sanity check first
    max_samples = dataset_cfg.get("max_samples")
    if max_samples:
        train_data = train_data.select(range(max_samples))
        logger.info(f"Limiting to {max_samples} samples for this run.")

    # --- Set up Groq client ---
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Copy .env.example to .env and fill it in.")
    client = Groq(api_key=api_key)

    # --- Output file setup ---
    output_path = Path(output_cfg["raw_traces"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Support resuming: skip already-processed questions
    processed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                record = json.loads(line)
                processed.add(record["question"])
        logger.info(f"Resuming: {len(processed)} traces already generated, skipping them.")

    # --- Rate limiting config ---
    # Groq free tier: ~30 requests/min. We add a small sleep between calls.
    # TODO: if you upgrade to Groq paid tier, increase requests_per_minute in distillation.yaml
    min_interval = 60.0 / teacher_cfg["requests_per_minute"]

    # --- Generate traces ---
    logger.info(f"Generating traces with teacher: {teacher_cfg['model']}")
    skipped, failed = 0, 0

    with open(output_path, "a") as out_f:
        for sample in tqdm(train_data, desc="Generating traces"):
            question = sample["question"]
            ground_truth = sample["answer"]  # includes #### at the end

            if question in processed:
                skipped += 1
                continue

            prompt = build_prompt(question, prompt_template)
            t0 = time.time()

            trace = call_teacher(
                client=client,
                prompt=prompt,
                model=teacher_cfg["model"],
                max_tokens=teacher_cfg["max_tokens"],
                temperature=teacher_cfg["temperature"],
            )

            if trace is None:
                failed += 1
                logger.warning(f"Skipping question due to API failure: {question[:60]}...")
                continue

            record = {
                "question": question,
                "ground_truth": ground_truth,
                "trace": trace,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()  # flush per line so partial runs are not lost

            # Respect rate limit
            elapsed = time.time() - t0
            sleep_time = max(0, min_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    logger.info(
        f"Done. Traces saved to {output_path}. "
        f"Skipped (already done): {skipped} | Failed: {failed}"
    )


if __name__ == "__main__":
    generate_traces()

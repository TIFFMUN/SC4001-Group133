"""
build_dataset.py
----------------
Converts teacher-generated JSONL traces into a HuggingFace Dataset
ready for SFTTrainer. Handles both D1 (all traces) and D2 (filtered traces).

Output format per sample:
    {"text": "<full prompt + response formatted for instruction tuning>"}

The dataset is saved to disk so notebooks can load it directly on Colab
without re-running the generation step.

Member responsible: A
"""

import json
import logging
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/distillation.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_finetune_config(config_path: str = "configs/finetune.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_sample(question: str, trace: str, tokenizer) -> str:
    """
    Format a question + CoT trace into the chat template expected by
    Qwen2.5-Instruct. Uses the tokenizer's built-in apply_chat_template.

    The student model is trained to reproduce the teacher's full reasoning
    trace given the question as input.

    TODO: experiment with different system prompts and observe if it affects
          how well the student reproduces reasoning structure.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Solve problems step by step.",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": trace},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_dataset(
    config_path: str = "configs/distillation.yaml",
    finetune_config_path: str = "configs/finetune.yaml",
    use_filtered: bool = False,
) -> None:
    """
    Main entry point.

    Args:
        use_filtered: If True, build dataset from D2 filtered traces.
                      If False, build from D1 all traces.
                      Controlled by finetune.yaml mode field at training time,
                      but can be overridden here for manual dataset construction.

    TODO: call this once for D1 (use_filtered=False) and once for D2 (use_filtered=True)
          so both datasets are ready before training begins.
    """
    cfg = load_config(config_path)
    ft_cfg = load_finetune_config(finetune_config_path)

    # Select input file based on D1 vs D2
    trace_path_key = "filtered_traces" if use_filtered else "raw_traces"
    trace_path = Path(cfg["output"][trace_path_key])

    if not trace_path.exists():
        raise FileNotFoundError(
            f"Traces not found at {trace_path}. "
            f"Run {'filter_traces.py' if use_filtered else 'generate_traces.py'} first."
        )

    output_dir = Path(cfg["output"]["dataset_dir"])
    dataset_name = "filtered" if use_filtered else "all"
    save_path = output_dir / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    # --- Load tokenizer ---
    model_name = ft_cfg["model"]["name"]
    logger.info(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Load and format traces ---
    logger.info(f"Loading traces from {trace_path}...")
    records = load_jsonl(trace_path)
    logger.info(f"Loaded {len(records)} traces.")

    formatted = [
        {"text": format_sample(r["question"], r["trace"], tokenizer)}
        for r in records
    ]

    # --- Train / val split ---
    split_cfg = cfg["splits"]
    train_ratio = split_cfg["train"]
    n_train = int(len(formatted) * train_ratio)

    train_samples = formatted[:n_train]
    val_samples = formatted[n_train:]

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_samples),
        "validation": Dataset.from_list(val_samples),
    })

    logger.info(
        f"Dataset split — train: {len(train_samples)}, val: {len(val_samples)}"
    )

    # --- Save to disk ---
    dataset_dict.save_to_disk(str(save_path))
    logger.info(f"Dataset saved to {save_path}")

    # TODO: upload save_path to Google Drive so Colab training notebook can load it
    # e.g. from google.colab import drive; drive.mount('/content/drive')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="Build D2 filtered dataset. Omit for D1 all-traces dataset.",
    )
    args = parser.parse_args()
    build_dataset(use_filtered=args.filtered)

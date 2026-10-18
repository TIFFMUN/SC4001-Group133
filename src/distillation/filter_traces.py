"""
filter_traces.py
----------------
Experiment: D2 (filtered distillation) — the primary novelty contribution.

Filters raw teacher traces by verifying that the teacher's final answer
matches the GSM8K ground truth. Only correct traces are kept for training.

Strategy D2 hypothesis:
    Training on verified-correct traces (higher quality data) should
    outperform training on all traces (D1), especially when the teacher
    makes occasional errors. The filter rate itself is a reportable metric.

Member responsible: A
"""

import re
import json
import logging
from pathlib import Path

import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/distillation.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_answer(text: str) -> str | None:
    """
    Extract the final numeric answer from a GSM8K-style response.

    GSM8K format uses '#### <number>' as the answer delimiter.
    We look for this pattern first, then fall back to the last number in the text.

    Returns the answer as a string (stripped, no commas) or None if not found.

    TODO: test this regex on a sample of raw traces before running the full filter
    TODO: consider handling answers with units (e.g. "36 apples") — currently ignored
    """
    # Primary: look for #### pattern (standard GSM8K format)
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback: last standalone number in the text
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    if numbers:
        return numbers[-1].strip()

    return None


def answers_match(predicted: str | None, ground_truth: str) -> bool:
    """
    Compare predicted answer to GSM8K ground truth.
    Both are extracted via extract_answer before comparison.
    """
    if predicted is None:
        return False

    gt_answer = extract_answer(ground_truth)
    if gt_answer is None:
        return False

    # Normalize: strip whitespace, remove trailing zeros after decimal
    try:
        return float(predicted) == float(gt_answer)
    except ValueError:
        return predicted.strip() == gt_answer.strip()


def filter_traces(config_path: str = "configs/distillation.yaml") -> None:
    """
    Main entry point.
    Reads raw_traces.jsonl, keeps only traces where the teacher's answer
    matches the GSM8K ground truth, writes to filtered_traces.jsonl.

    Logs the filter rate (% kept) — report this in the paper as evidence
    of teacher quality and as motivation for D2 over D1.
    """
    cfg = load_config(config_path)
    raw_path = Path(cfg["output"]["raw_traces"])
    filtered_path = Path(cfg["output"]["filtered_traces"])
    filtered_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw traces not found at {raw_path}. Run generate_traces.py first."
        )

    total, kept, no_answer = 0, 0, 0

    with open(raw_path) as in_f, open(filtered_path, "w") as out_f:
        lines = in_f.readlines()
        for line in tqdm(lines, desc="Filtering traces"):
            record = json.loads(line)
            total += 1

            predicted = extract_answer(record["trace"])
            if predicted is None:
                no_answer += 1
                continue

            if answers_match(predicted, record["ground_truth"]):
                out_f.write(json.dumps(record) + "\n")
                kept += 1

    filter_rate = kept / total if total > 0 else 0
    logger.info(
        f"Filter complete: {kept}/{total} traces kept ({filter_rate:.1%}). "
        f"No answer found: {no_answer}. "
        f"Filtered traces saved to {filtered_path}."
    )

    # TODO: log this filter rate to results/ for inclusion in the report table
    # TODO: investigate what % of dropped traces had a wrong answer vs no answer at all


if __name__ == "__main__":
    filter_traces()

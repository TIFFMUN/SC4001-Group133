"""
budget_router.py
----------------
Inference-gated budget router for Exp 6.

Routes each GSM8K question to one of two generation profiles based on an
estimated difficulty score computed with a single forward pass:

  complexity = (input_len * 0.1) + (-log_prob_of_last_token * 0.5)

  score < threshold  →  "easy" profile  (low max_new_tokens, greedy)
  score >= threshold →  "hard" profile  (high max_new_tokens, greedy)

This is purely an inference-time mechanism — no extra training required.
Run only after Exps 0–5 are complete and the best checkpoint is available.
"""

import time
import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    prediction: str
    token_count: int
    latency_ms: float
    route: str        # "easy" or "hard"
    complexity: float


def compute_complexity(
    model: AutoModelForCausalLM,
    inputs: dict,
) -> float:
    """
    Estimate prompt difficulty with one forward pass — no generation.

    Uses mean NLL of the prompt under the model (same pattern as eval_arc.py's
    score_choice) combined with prompt length. Higher score = harder question.
    """
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    # outputs.loss is mean NLL per token; longer + higher NLL = harder
    mean_nll = outputs.loss.item()
    return input_len * 0.1 + mean_nll * 0.5


class BudgetRouter:
    """
    Routes generation to a fast or budget profile based on prompt complexity.

    Args:
        model:          merged (or base) causal LM, already on device
        tokenizer:      matching tokenizer
        easy_kwargs:    kwargs forwarded to model.generate() for easy prompts
        hard_kwargs:    kwargs forwarded to model.generate() for hard prompts
        threshold:      complexity score cutoff; above this → hard profile
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        easy_kwargs: dict,
        hard_kwargs: dict,
        threshold: float,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.easy_kwargs = easy_kwargs
        self.hard_kwargs = hard_kwargs
        self.threshold = threshold

    def route(self, prompt: str) -> RouteResult:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        complexity = compute_complexity(self.model, inputs)

        if complexity < self.threshold:
            route = "easy"
            gen_kwargs = self.easy_kwargs
        else:
            route = "hard"
            gen_kwargs = self.hard_kwargs

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                pad_token_id=self.model.config.eos_token_id,
                **gen_kwargs,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        generated_ids = output_ids[0][input_len:]
        prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_count = len(generated_ids)

        return RouteResult(
            prediction=prediction,
            token_count=token_count,
            latency_ms=latency_ms,
            route=route,
            complexity=complexity,
        )

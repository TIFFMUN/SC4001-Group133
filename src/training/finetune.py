"""
finetune.py
-----------
LoRA fine-tuning using HuggingFace SFTTrainer + PEFT.

Handles all three training experiments via finetune.yaml:
  Exp 2 (A1): adaptation   — trains on GSM8K human-labelled data
  Exp 3 (D1): distillation — trains on all teacher CoT traces
  Exp 4 (D2): filtered     — trains on answer-verified teacher traces

Ablations (change ONE field in finetune.yaml, re-run):
  A2: lora.rank ∈ {4, 8, 16, 32}
  A3: data.train_fraction ∈ {0.1, 0.25, 0.5, 1.0}

Member responsible: B
"""

import logging
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

from src.training.utils import load_config, set_seed, save_run_config, get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data loading
# =============================================================================

def load_adaptation_data(cfg: dict) -> tuple:
    """
    Load GSM8K human-labelled training data for Exp 2 (A1 adaptation).

    Formats each sample as an instruction-tuning conversation:
        user:      the math question
        assistant: the ground truth solution (includes #### answer)
    """
    dataset = load_dataset("openai/gsm8k", "main")
    train_data = dataset["train"]

    # A3 ablation: subsample training data
    # TODO: set data.train_fraction in finetune.yaml to 0.1/0.25/0.5/1.0 for A3 sweep
    fraction = cfg["data"].get("train_fraction", 1.0)
    if fraction < 1.0:
        n = int(len(train_data) * fraction)
        train_data = train_data.select(range(n))
        logger.info(f"A3 ablation: using {fraction:.0%} of training data ({n} samples)")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    def format_sample(sample: dict) -> dict:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful math tutor. Solve problems step by step.",
            },
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    train_data = train_data.map(format_sample, remove_columns=train_data.column_names)

    # Use 10% of training data as validation
    split = train_data.train_test_split(test_size=0.1, seed=cfg["training"]["seed"])
    return split["train"], split["test"]


def load_distillation_data(cfg: dict, use_filtered: bool) -> tuple:
    """
    Load teacher-generated CoT traces for Exp 3 (D1) or Exp 4 (D2).

    Args:
        use_filtered: True for D2 (filtered traces), False for D1 (all traces)
    """
    dataset_name = "filtered" if use_filtered else "all"
    dataset_dir = Path(cfg["data"]["distillation_dataset_dir"]) / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Distillation dataset not found at {dataset_dir}. "
            f"Run src/distillation/build_dataset.py {'--filtered' if use_filtered else ''} first."
        )

    dataset = load_from_disk(str(dataset_dir))

    # A3 ablation: subsample distillation data
    fraction = cfg["data"].get("train_fraction", 1.0)
    if fraction < 1.0:
        n = int(len(dataset["train"]) * fraction)
        dataset["train"] = dataset["train"].select(range(n))
        logger.info(f"A3 ablation: using {fraction:.0%} of distillation data ({n} samples)")

    return dataset["train"], dataset["validation"]


# =============================================================================
# Model setup
# =============================================================================

def load_model_and_tokenizer(cfg: dict):
    """Load base model and tokenizer. Apply 4-bit/8-bit quantization if configured."""
    model_name = cfg["model"]["name"]
    load_in_4bit = cfg["model"].get("load_in_4bit", False)
    load_in_8bit = cfg["model"].get("load_in_8bit", False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config — useful if Colab T4 runs out of VRAM
    # TODO: if you get CUDA OOM errors, set load_in_4bit: true in finetune.yaml
    #       and install bitsandbytes: pip install bitsandbytes
    bnb_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if not (load_in_4bit or load_in_8bit) else None,
        device_map="auto",
    )

    return model, tokenizer


def build_lora_config(cfg: dict) -> LoraConfig:
    """
    Build PEFT LoRA config from finetune.yaml.

    A2 ablation: change lora.rank in finetune.yaml and re-run.
    Rule of thumb: keep lora.alpha = 2 * lora.rank.

    TODO: for A2 ablation, document each run clearly:
          rank=4  → save to results/checkpoints/lora_r4/
          rank=8  → save to results/checkpoints/lora_r8/
          etc.
    """
    lora_cfg = cfg["lora"]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )


# =============================================================================
# Training
# =============================================================================

def train(config_path: str = "configs/finetune.yaml") -> None:
    """
    Main entry point. Loads config, data, model, and runs SFTTrainer.

    To switch experiments, change `data.mode` in finetune.yaml:
        "adaptation"   → Exp 2
        "distillation" → Exp 3
        "filtered"     → Exp 4
    """
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])
    device = get_device()
    logger.info(f"Training device: {device}")

    # --- Load data based on mode ---
    mode = cfg["data"]["mode"]
    logger.info(f"Training mode: {mode}")

    if mode == "adaptation":
        train_data, val_data = load_adaptation_data(cfg)
    elif mode == "distillation":
        train_data, val_data = load_distillation_data(cfg, use_filtered=False)
    elif mode == "filtered":
        train_data, val_data = load_distillation_data(cfg, use_filtered=True)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: adaptation, distillation, filtered.")

    logger.info(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

    # --- Load model + tokenizer ---
    model, tokenizer = load_model_and_tokenizer(cfg)
    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training arguments ---
    train_cfg = cfg["training"]
    training_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_seq_length=train_cfg["max_seq_length"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        fp16=train_cfg["fp16"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",   # set to "wandb" if you want experiment tracking
        seed=train_cfg["seed"],
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    # Save config alongside checkpoint for reproducibility
    save_run_config(cfg, Path(train_cfg["output_dir"]))

    logger.info("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    logger.info(f"Model saved to {train_cfg['output_dir']}")

    # TODO: copy output_dir to Google Drive after training completes
    # Example: !cp -r {train_cfg['output_dir']} /content/drive/MyDrive/SC4001/results/checkpoints/


# =============================================================================
# OPTIONAL EXTENSION: GRPO (Group Relative Policy Optimization)
# Based on DeepSeek-R1 (2025) — rule-based RL for math reasoning
# Reference: https://arxiv.org/abs/2501.12599
# TRL docs:  https://huggingface.co/docs/trl/grpo_trainer
#
# Only attempt this after Exp 2 (A1 LoRA) is fully working and evaluated.
# If GRPO works: strong novelty angle vs LoRA.
# If unstable:   document instability as a finding (negative results are valid).
#
# TODO (optional): implement reward function
#   def reward_fn(completions: list[str], ground_truths: list[str]) -> list[float]:
#       from src.distillation.filter_traces import extract_answer, answers_match
#       rewards = []
#       for completion, gt in zip(completions, ground_truths):
#           predicted = extract_answer(completion)
#           if answers_match(predicted, gt):
#               rewards.append(1.0)   # correct answer
#           elif predicted is not None:
#               rewards.append(0.1)   # found a number but wrong
#           else:
#               rewards.append(0.0)   # no answer found
#       return rewards
#
# TODO (optional): implement train_grpo()
#   from trl import GRPOTrainer, GRPOConfig
#   Set G=8 rollouts per question (fits on T4 with 0.5B model)
#   Start KL penalty at 0.01 and tune if training collapses
#   Log reward curve per step to compare convergence vs LoRA
# =============================================================================


if __name__ == "__main__":
    train()

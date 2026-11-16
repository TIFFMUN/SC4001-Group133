# SC4001 Group 133 — Efficient Reasoning with Small Language Models

**Project G** | SC4001 Neural Networks and Deep Learning | NTU

## Research Question

Can targeted reasoning-chain distillation from a larger model improve a small model's accuracy on math and commonsense benchmarks more efficiently than fine-tuning on human-annotated data alone, and does the improvement transfer across task types?

## Models

| Role | Model |
|---|---|
| Student | `Qwen/Qwen2.5-0.5B-Instruct` |
| Teacher | `Qwen/Qwen2.5-32B` via Groq API |

## Key Findings & Final Results

Our experimental investigation followed a two-stage evaluation methodology:
1. **Phase 1 Pilot Screening ($N=100$, 256 tokens):** Initial runs showed baseline at 29% and LoRA at 35%. However, token diagnostics revealed that **68% of baseline generations hit the 256-token ceiling**, causing mid-derivation truncation.
2. **Phase 2 Full-Scale Benchmark ($N=500+$, 600 tokens):** Expanding the budget to 600 tokens reduced truncation to 1.4%, revealing true zero-shot base capability (46.00%). LoRA fine-tuning on concise human labels induced severe **length collapse** (112.9 tokens vs 304.3 base tokens, 31.20%), while **teacher distillation restored the reasoning scratchpad** (~400 tokens, 40.80% in Exp 3 and 35.80% in Exp 4).

| Exp | Strategy | Supervision / Checkpoint | Benchmark | Samples ($N$) | Max Tokens | Accuracy | Mean Tokens | Latency |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **0** | Zero-Shot Base | Pre-trained `Qwen2.5-0.5B` | GSM8K | 500 | 600 | **46.00%** | 304.3 | 12.69s |
| **1** | Few-Shot CoT | 4 In-Context Exemplars | GSM8K | 100 | 256 | **23.00%** | 236.8 | 10.78s |
| **2** | LoRA Adaptation | Human GSM8K SFT ($r=16$) | GSM8K | 500 | 600 | **31.20%** | **112.9** | **4.72s** |
| **3** | Distillation (D1) | All 522 Teacher Rationales | GSM8K | 500 | 600 | **40.80%** | 407.7 | 11.66s |
| **4** | Distillation (D2) | 408 Filtered Rationales | GSM8K | 500 | 600 | **35.80%** | 391.1 | 11.49s |
| **5** | Cross-Task Transfer | Exp 3 Distilled Checkpoint | ARC-Challenge | **1,172 (Full)** | Log-prob | **33.62%** | 25.0 | **0.13s** |
| **6** | Inference Router | Complexity Gating ($S(x)$) | GSM8K | 100 | 128 / 600 | **Pareto Curve** | 128–266 | 5.75–12.04s |
| **LoRA-Distill** | LoRA on Student | Filtered Student + LoRA | GSM8K | 100 | 600 | **15.00%** | 181.9 | 7.72s |

## Repo Structure

```
configs/          YAML configs for distillation, fine-tuning, and evaluation
data/             Dataset instructions (auto-downloaded via HuggingFace)
src/
  distillation/   Teacher trace generation, filtering, dataset construction
  training/       LoRA fine-tuning (SFTTrainer + PEFT)
  baselines/      Zero-shot and few-shot CoT prompting
  evaluation/     GSM8K and ARC evaluation scripts + metrics
notebooks/        Thin Colab-facing notebooks (import from src/)
results/          Output CSVs and plots (gitignored, store on Drive)
Dockerfile        CPU inference/eval only
```

## Setup

### Local / Docker (eval only)

```bash
git clone <this-repo> && cd SC4001-Group133
cp .env.example .env          # fill in GROQ_API_KEY
pip install -r requirements.txt
```

### Google Colab (training)

Open the relevant notebook in `notebooks/` and follow the steps inside.
Each notebook handles Drive mounting, repo cloning, and dependency installation.

## Reproducing Results

### Step 1 — Generate distillation data (Member A)
```bash
# Set GROQ_API_KEY in .env first
python -m src.distillation.generate_traces
python -m src.distillation.filter_traces
python -m src.distillation.build_dataset           # D1 all traces
python -m src.distillation.build_dataset --filtered  # D2 filtered traces
```

### Step 2 — Fine-tune student (Member B, on Colab)
Edit `configs/finetune.yaml` to set `data.mode`, then run notebook `02_finetune_student.ipynb`.

| Experiment | `data.mode` | `training.output_dir` |
|---|---|---|
| Exp 2 | `adaptation` | `results/checkpoints/adaptation` |
| Exp 3 | `distillation` | `results/checkpoints/distillation` |
| Exp 4 | `filtered` | `results/checkpoints/filtered` |

### Step 3 — Evaluate (Member C)

#### Phase 1: 100-Sample Quick Screening (256-token limit)
```bash
# Evaluate GSM8K Zero-Shot (Exp 0)
python -m src.evaluation.eval_gsm8k --experiment exp0_zero_shot --max_samples 100

# Evaluate GSM8K 4-Shot CoT (Exp 1)
python -m src.evaluation.eval_gsm8k --experiment exp1_few_shot --few_shot --max_samples 100

# Evaluate GSM8K LoRA Adaptation (Exp 2)
python -m src.evaluation.eval_gsm8k --experiment exp2_adaptation_r16 --checkpoint results/checkpoints/lora_r16 --max_samples 100

# Evaluate Inference Budget Router Sweep (Exp 6)
python -m src.evaluation.eval_router --experiment exp6_router --checkpoint results/checkpoints/lora_r16 --max_samples 100
```

#### Phase 2: Full-Scale Benchmark (600-token budget, 500+ samples)
```bash
# Full GSM8K Zero-Shot Base (Exp 0, 500 samples, 600 tokens)
python -m src.evaluation.eval_gsm8k --experiment exp0_zero_shot_500 --max_samples 500

# Full GSM8K LoRA Adaptation (Exp 2, 500 samples, 600 tokens)
python -m src.evaluation.eval_gsm8k --experiment exp2_adaptation_r16_500 --checkpoint results/checkpoints/lora_r16 --max_samples 500

# Full GSM8K Distillation All Traces (Exp 3, 500 samples, 600 tokens)
python -m src.evaluation.eval_gsm8k --experiment exp3_distillation_500 --checkpoint results/checkpoints/distillation --max_samples 500

# Full GSM8K Distillation Filtered Traces (Exp 4, 500 samples, 600 tokens)
python -m src.evaluation.eval_gsm8k --experiment exp4_filtered_500 --checkpoint results/checkpoints/filtered --max_samples 500

# Full ARC-Challenge Cross-Task Transfer (Exp 5, all 1,172 test samples)
python -m src.evaluation.eval_arc --experiment exp5_arc_challenge --checkpoint results/checkpoints/distillation
```

Or run `notebooks/03_evaluate_updaredall.ipynb` on Google Colab or Kaggle.

### Step 4 — Plots
Run `notebooks/04_analysis_plots.ipynb` after all CSVs are in `results/`.

### Docker (reproducible eval)
```bash
docker build -t sc4001 .
docker run --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/results:/app/results \
  --env-file .env \
  sc4001 python -m src.evaluation.eval_gsm8k --experiment exp0_zero_shot
```

## Dependencies

See `requirements.txt`. Key packages:
- `transformers`, `peft`, `trl`, `datasets`, `accelerate` — HuggingFace stack
- `torch` — PyTorch backend
- `groq` — teacher API calls
- `pandas`, `matplotlib`, `seaborn` — results analysis

Python 3.11 recommended.

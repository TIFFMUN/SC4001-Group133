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

**Recommended: Use the Jupyter notebooks on Google Colab or Kaggle** (they handle dependencies, Drive mounting, and API configuration automatically).

### Notebook Workflow (Recommended)

1. **Notebook 01** — `notebooks/01_generate_distillation_data.ipynb` (Member A)
   - Generates teacher-rationale traces using Groq API
   - Filters traces by answer correctness
   - Builds D1 (unfiltered) and D2 (filtered) datasets

2. **Notebook 02** — `notebooks/02_finetune_student.ipynb` (Member B)
   - Fine-tunes Qwen2.5-0.5B using LoRA on three supervision sources:
     - Exp 2: Human-annotated GSM8K solutions
     - Exp 3: Unfiltered teacher rationales (D1)
     - Exp 4: Filtered teacher rationales (D2)

3. **Notebook 03** — `notebooks/03_evaluate_all.ipynb` (Member C)
   - Evaluates all experiments (Exp 0–6) on GSM8K and ARC-Challenge
   - Generates per-experiment result CSVs
   - Implements the inference-gated budget router (Exp 6)

4. **Notebook 04** — `notebooks/04_analysis_plots.ipynb`
   - Loads result CSVs and generates comparison plots
   - Computes Pareto frontier for Exp 6

### Local Command-Line (Advanced)

If running locally (requires HuggingFace login and Groq API key):

```bash
git clone <this-repo> && cd SC4001-Group133
pip install -r requirements.txt
export HF_TOKEN=<your-huggingface-token>
export GROQ_API_KEY=<your-groq-api-key>

# Generate distillation data
python -m src.distillation.generate_traces
python -m src.distillation.filter_traces

# Fine-tune (Exp 2, 3, 4)
# Edit configs/finetune.yaml to set data.mode
python -m src.training.finetune

# Evaluate
python -m src.evaluation.eval_gsm8k --experiment exp0_zero_shot --max_samples 500
python -m src.evaluation.eval_router --experiment exp6_router --max_samples 30
```

Note: Python script execution may require additional environment setup. Notebooks are preferred.

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

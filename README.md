# SC4001 Group 133 — Efficient Reasoning with Small Language Models

**Project G** | SC4001 Neural Networks and Deep Learning | NTU

## Research Question

Can targeted reasoning-chain distillation from a larger model improve a small model's accuracy on math and commonsense benchmarks more efficiently than fine-tuning on human-annotated data alone, and does the improvement transfer across task types?

## Models

| Role | Model |
|---|---|
| Student | `Qwen/Qwen2.5-0.5B-Instruct` |
| Teacher | `Qwen/Qwen2.5-7B-Instruct` via Groq API |

## Experiments

| Exp | Strategy | Description | Benchmark | Status / Result |
|---|---|---|---|---|
| 0 | Baseline | Zero-shot prompting | GSM8K | **29.00%** |
| 1 | Baseline | Few-shot CoT prompting (4 shots) | GSM8K | **23.00%** |
| 2 | Adaptation (A1) | LoRA fine-tune on GSM8K human labels ($r=16$) | GSM8K | **35.00%** |
| 3 | Distillation (D1) | LoRA fine-tune on all teacher CoT traces | GSM8K | Pending |
| 4 | Distillation (D2) | LoRA fine-tune on filtered (correct) traces | GSM8K | Pending |
| 5 | Transfer (D3) | Exp 4 model evaluated on ARC — cross-task transfer | ARC | Pending |
| 6 | Inference Router | Difficulty-gated budget routing (Pareto curve) | GSM8K | **35.00% → 3.00%** |

**Ablations:** A2 (LoRA rank ∈ {4,8,16,32}), A3 (data size ∈ {10,25,50,100%})

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
Edit `configs/eval.yaml` to set `model.checkpoint_path`, then:
```bash
python -m src.evaluation.eval_gsm8k --experiment exp2_adaptation_r16
python -m src.evaluation.eval_arc   --experiment exp5_d3_cross_task
```

Or run `notebooks/03_evaluate_all.ipynb` on Colab.

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

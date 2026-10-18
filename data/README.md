# Data

All datasets are loaded automatically via the HuggingFace `datasets` library.
No manual download is required.

## Datasets Used

### GSM8K (Grade School Math)
- **Source:** `load_dataset("openai/gsm8k", "main")`
- **Train:** 7,473 samples | **Test:** 1,319 samples
- **Used for:** Exp 0-5 (primary benchmark), Exp 2 adaptation training, Exp 3/4 distillation training
- **License:** MIT

### ARC-Challenge (AI2 Reasoning Challenge)
- **Source:** `load_dataset("allenai/ai2_arc", "ARC-Challenge")`
- **Train:** 1,119 samples | **Test:** 1,172 samples
- **Used for:** Exp 5 (D3) cross-task transfer evaluation only — never used for training
- **License:** CC BY-SA 4.0

## Generated Data (gitignored)

The following directories are created at runtime and are NOT committed to git.
Store these on Google Drive for persistence across Colab sessions.

```
data/traces/raw_traces.jsonl       # All teacher CoT traces (before filtering)
data/traces/filtered_traces.jsonl  # D2: answer-verified traces only
data/processed/                    # Final HuggingFace Dataset (train/val splits)
```

## Quick Load

```python
from datasets import load_dataset

gsm8k = load_dataset("openai/gsm8k", "main")
arc   = load_dataset("allenai/ai2_arc", "ARC-Challenge")

# Access splits
gsm8k_train = gsm8k["train"]   # 7473 samples
gsm8k_test  = gsm8k["test"]    # 1319 samples — keep separate until final eval
arc_test    = arc["test"]      # 1172 samples
```

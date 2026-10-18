# =============================================================================
# Dockerfile — CPU inference and evaluation only
# Training is done on Google Colab with GPU (see notebooks/02_finetune_student.ipynb)
#
# Usage:
#   Build:  docker build -t sc4001 .
#   Run:    docker run --rm \
#             -v ~/.cache/huggingface:/root/.cache/huggingface \
#             -v $(pwd)/results:/app/results \
#             --env-file .env \
#             sc4001 python -m src.evaluation.eval_gsm8k --experiment exp0_zero_shot
#
# The HuggingFace cache volume mount means datasets and models are downloaded
# once and reused across container runs.
# =============================================================================

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# torch CPU-only build — significantly smaller image than CUDA build
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/

# Results directory — mount as volume to persist outputs outside the container
RUN mkdir -p results

# Default command: run zero-shot baseline on GSM8K
# Override at runtime: docker run sc4001 python -m src.evaluation.eval_gsm8k --experiment exp2_adaptation
CMD ["python", "-m", "src.evaluation.eval_gsm8k", "--experiment", "exp0_zero_shot"]

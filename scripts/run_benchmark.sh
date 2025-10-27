#!/usr/bin/env bash
# Example usage script for macOS
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

# activate venv if it exists
if [ -d "$BASE_DIR/venv_indic" ]; then
  echo "Activating venv_indic..."
  source "$BASE_DIR/venv_indic/bin/activate"
fi

# Example: benchmark using the repo's inference script (adjust args to your model)
# Replace the command below with the actual CLI you use for inference.
CMD='python inference/custom_interactive.py --model path/to/model.pt'

# Run benchmark: 200 iterations, 10 warmup, concurrency 1
python "$BASE_DIR/scripts/benchmark.py" --mode cmd --cmd "$CMD" --input-file "$BASE_DIR/scripts/sample_sentences.txt" --iters 200 --warmup 10 --concurrency 1

# If you have a python callable to call directly (module:function), example:
# python "$BASE_DIR/scripts/benchmark.py" --mode python --callable "huggingface_interface.example:translate" --input-file "$BASE_DIR/scripts/sample_sentences.txt" --iters 200
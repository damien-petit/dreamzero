#!/bin/bash
# Run test_wan22_standalone.py on all three models sequentially (fresh process each).
set -e
export PATH=/home/d/miniconda3/envs/dreamzero/bin:$PATH
export TORCH_COMPILE_DISABLE=1
cd /home/d/devel-src/dreamzero

declare -A MODELS=(
  ["wan21_14B_pretrained"]="./checkpoints/DreamZero-DROID"
  ["wan22_r4_100steps"]="./checkpoints/dreamzero_droid_wan22_lora"
  ["wan22_r32_5000steps"]="./checkpoints/dreamzero_droid_wan22_lora_r32/checkpoint-5000"
)

for label in wan21_14B_pretrained wan22_r4_100steps wan22_r32_5000steps; do
  path="${MODELS[$label]}"
  echo "=================================================="
  echo "RUNNING STANDALONE: $label  ($path)"
  echo "=================================================="
  python test_wan22_standalone.py \
    --model_path "$path" \
    --output_dir "./checkpoints/standalone_compare/$label" \
    || echo "FAILED: $label"
done
echo "ALL DONE"

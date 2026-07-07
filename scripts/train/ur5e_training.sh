#!/bin/bash
# DreamZero UR5e Training Script
#
# Supports both Wan2.2-TI2V-5B (5B, default) and Wan2.1-I2V-14B (14B) backbones.
#
# Usage:
#   # Wan2.2 5B — single GPU (default)
#   DATA_ROOT=./data/ur5e_dataset bash scripts/train/ur5e_training.sh
#
#   # Wan2.1 14B — multi-GPU
#   BACKBONE=wan21 NUM_GPUS=2 DATA_ROOT=./data/ur5e_dataset bash scripts/train/ur5e_training.sh
#
# Prerequisites:
#   - UR5e dataset recorded and converted with convert_lerobot_to_gear.py at DATA_ROOT
#   - For wan22: Wan2.2-TI2V-5B weights + Wan2.1-I2V-14B-480P (CLIP only)
#     huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B
#     huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
#   - For wan21: Wan2.1-I2V-14B-480P weights
#     huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P

export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -n "$DREAMZERO_ROOT" ] && [ -d "$DREAMZERO_ROOT/groot" ]; then
    :
elif [ -d "/root/yejink/dreamzero/groot" ]; then
    DREAMZERO_ROOT=/root/yejink/dreamzero
elif [ -d "/root/dreamzero/groot" ]; then
    DREAMZERO_ROOT=/root/dreamzero
elif [ -d "$SCRIPT_REPO_ROOT/groot" ]; then
    DREAMZERO_ROOT="$SCRIPT_REPO_ROOT"
else
    DREAMZERO_ROOT="${DREAMZERO_ROOT:-/root/yejink/dreamzero}"
fi
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo "ERROR: No groot/ under $DREAMZERO_ROOT. Set DREAMZERO_ROOT to the dreamzero repo root."
    exit 1
fi

# ============ BACKBONE SELECTION ============
BACKBONE=${BACKBONE:-wan22}

if [ "$BACKBONE" = "wan22" ]; then
    ACTION_HEAD=wan_flow_matching_action_tf_wan22
    WAN_CKPT_DIR=${WAN22_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B"}
    # CLIP encoder is not bundled in Wan2.2 — use from Wan2.1
    IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
    VAE_PATH="$WAN_CKPT_DIR/Wan2.2_VAE.pth"
    T5_PATH="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
    IMG_H=160
    IMG_W=320
    FRAME_SEQLEN=50
    DEFAULT_OUTPUT="$DREAMZERO_ROOT/checkpoints/dreamzero_ur5e_wan22_lora"
elif [ "$BACKBONE" = "wan21" ]; then
    ACTION_HEAD=wan_flow_matching_action_tf
    WAN_CKPT_DIR=${WAN21_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
    IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"$WAN_CKPT_DIR"}
    VAE_PATH="$WAN_CKPT_DIR/Wan2.1_VAE.pth"
    T5_PATH="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
    IMG_H=176
    IMG_W=320
    FRAME_SEQLEN=55
    DEFAULT_OUTPUT="$DREAMZERO_ROOT/checkpoints/dreamzero_ur5e_wan21_lora"
else
    echo "ERROR: Unknown BACKBONE='$BACKBONE'. Use 'wan22' or 'wan21'."
    exit 1
fi

echo "Backbone: $BACKBONE  (action_head=$ACTION_HEAD, ${IMG_H}x${IMG_W})"

# ============ USER CONFIGURATION ============
NUM_GPUS=${NUM_GPUS:-1}
DATA_ROOT=${DATA_ROOT:?"ERROR: Set DATA_ROOT to the converted UR5e dataset path."}
OUTPUT_DIR=${OUTPUT_DIR:-"$DEFAULT_OUTPUT"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$DREAMZERO_ROOT/checkpoints/umt5-xxl"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ "$BACKBONE" = "wan22" ]; then
    if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
        echo "Wan2.2-TI2V-5B not found at $WAN_CKPT_DIR. Downloading..."
        huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN_CKPT_DIR"
    fi
fi
if [ ! -d "$IMAGE_ENCODER_DIR" ] || [ ! -f "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "CLIP encoder not found at $IMAGE_ENCODER_DIR. Downloading Wan2.1-I2V-14B-480P..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$IMAGE_ENCODER_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found. Downloading..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
# ================================================

# ============ VALIDATE DATASET ============
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATA_ROOT"
    exit 1
fi
if [ ! -f "$DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: meta/embodiment.json missing — run convert_lerobot_to_gear.py first:"
    echo "  python scripts/data/convert_lerobot_to_gear.py --dataset-path $DATA_ROOT --embodiment-tag ur5e \\"
    echo "      --state-keys '{\"joint_position\": [0, 6], \"gripper_position\": [6, 7]}' \\"
    echo "      --action-keys '{\"joint_position\": [0, 6], \"gripper_position\": [6, 7]}' \\"
    echo "      --relative-action-keys joint_position --task-key annotation.task"
    exit 1
fi
# ==========================================

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"
if [ ! -f "$EXPERIMENT_PY" ]; then
    echo "ERROR: Not found: $EXPERIMENT_PY"
    exit 1
fi

PYTHON_311="/usr/bin/python3.11"
if [ -x "$PYTHON_311" ]; then
    RUN_CMD=( "$PYTHON_311" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" )
    echo "Using Python 3.11: $PYTHON_311"
else
    RUN_CMD=( python3 -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" )
    echo "Using: $(command -v python3)"
fi

cd "$DREAMZERO_ROOT"

"${RUN_CMD[@]}" \
    report_to=wandb \
    data=dreamzero/ur5e_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=$ACTION_HEAD \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=5000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    save_lora_only=true \
    max_chunk_size=4 \
    save_strategy=steps \
    image_resolution_height=$IMG_H \
    image_resolution_width=$IMG_W \
    frame_seqlen=$FRAME_SEQLEN \
    ur5e_data_root=$DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$T5_PATH \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$VAE_PATH \
    tokenizer_path=$TOKENIZER_DIR

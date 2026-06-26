#!/bin/bash
# Launch DreamZero WAN22 server using the cleaner serve_dreamzero_wan22.py script.
# Uses the eval_utils server which properly resets current_start_frame and wraps
# VAE decode in torch.no_grad().
#
# Usage:
#   conda activate dreamzero
#   bash launch_server_WAN22_v2.sh
#
# Then in a separate terminal:
#   conda activate dreamzero
#   python test_client_AR.py --port 5000
#
# Generated videos are saved to:
#   ./checkpoints/real_world_eval_gen_YYYYMMDD_0/dreamzero_droid_wan22_lora/

DATE_SUFFIX=$(date +%Y%m%d)
VIDEO_OUTPUT_DIR="./checkpoints/real_world_eval_gen_${DATE_SUFFIX}_0"

TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python eval_utils/serve_dreamzero_wan22.py \
    --model_path ./checkpoints/dreamzero_droid_wan22_lora \
    --port 5000 \
    --save_video_pred \
    --video_output_dir "${VIDEO_OUTPUT_DIR}"

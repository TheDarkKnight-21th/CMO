export WANDB_DISABLED=false # false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online # online


export HF_HOME="/data2/hg_models"
# export CUDA_VISIBLE_DEVICES="0,1,2,3"

# bash scripts/single_node/grpo_fast.sh

# 1 GPU
# export CUDA_VISIBLE_DEVICES="0"
# accelerate launch --num_processes=1 --main_process_port 29501 scripts/train_sd3_fast_ours.py --config config/grpo.py:ours_sd3_fast_1gpu    

# 4 GPU
export CUDA_VISIBLE_DEVICES="0,1,2,3"
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29521 scripts/train_sd3_fast_ours.py --config config/grpo.py:ours_sd3_fast_4gpu

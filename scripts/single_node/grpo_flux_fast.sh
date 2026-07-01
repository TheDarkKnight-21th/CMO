# 8 GPU
export WANDB_DISABLED=false # false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online # online
export HF_HOME="/data2/hg_models"

# bash scripts/single_node/grpo_flux_fast.sh

# export CUDA_VISIBLE_DEVICES="5"
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=1 --main_process_port 28401 scripts/train_flux_fast_ours.py --config config/grpo.py:ours_flux_fast_1gpu

export CUDA_VISIBLE_DEVICES="0,1,2,3"
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=4 --main_process_port 28501 scripts/train_flux_fast.py --config config/grpo.py:conceptmix_flux_fast_4gpu
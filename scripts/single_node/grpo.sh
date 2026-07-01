export HF_HOME="/data2/hg_models"

# 1 GPU
export CUDA_VISIBLE_DEVICES="1"
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29401 scripts/train_sd3.py --config config/grpo.py:geneval_sd3_1gpu
# 4 GPU
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:geneval_sd3

export HF_HOME="/data2/hg_models"
export CUDA_VISIBLE_DEVICES="7"

# python question.py \
#     --config "3000_refined_config_k=7.json" \
#     --sentence "3000_sentences_k=7.json" \
#     --output "3000_sentences_v2_k=7.json" \
#     --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
#     --tp 1

export CUDA_VISIBLE_DEVICES="6"
python question.py \
    --config "final_config_all.json" \
    --sentence "final_sentences_all.json" \
    --output "final_sentences_v2_all.json" \
    --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --tp 1
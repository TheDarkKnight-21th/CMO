# export HF_HOME="/data1/jungmyungwi/.cache/huggingface"
# gunicorn "app_conceptmix:create_app()"

# export HF_HOME='/data2/hg_models'
# gunicorn "app_geneval:create_app()"

export HF_HOME='/data2/hg_models'
gunicorn "app_ours:create_app()"
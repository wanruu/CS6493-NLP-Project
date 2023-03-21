export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATE=`date +%Y%m%d_%H_%M`
SAVEGEN=res/${DATE}
# --model_name_or_path="./gpt2-civil"\

# deepspeed --num_gpus=4 run.py \
# --deepspeed zero2.json \
#  python -u run.py \
python run_t5_NQG.py \
    --save_dir ${SAVEGEN}.csv



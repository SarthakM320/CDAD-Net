PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

# hostname
# nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
SAVE_DIR=/Users/sarthakm/Desktop/biplabsir_research/gcd/outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

python -m methods.contrastive_training.contrastive_training \
            --dataset_name 'officehome' \
            --batch_size 8 \
            --grad_from_block 11 \
            --epochs 100 \
            --base_model vit_dino \
            --num_workers 16 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v1' 'v2' \
> ${SAVE_DIR}logfile_${EXP_NUM}.out
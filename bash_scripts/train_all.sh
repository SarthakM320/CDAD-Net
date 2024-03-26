# PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'
export CUDA_VISIBLE_DEVICES=0
SAVE_DIR=outputs/

# for split in {13..15}; do
splits=('5_3' '5_5' '5_10')
for split in "${splits[@]}"; do
    EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
    EXP_NUM=$((EXP_NUM + 1))

    echo "Running experiment $EXP_NUM with split $split"

    python -m methods.contrastive_training.contrastive_training \
        --dataset_name 'officehome' \
        --batch_size 128 \
        --grad_from_block 11 \
        --epochs 75 \
        --base_model vit_dino \
        --num_workers 16 \
        --use_ssb_splits 'True' \
        --sup_con_weight 0.35 \
        --weight_decay 5e-5 \
        --contrast_unlabel_only 'False' \
        --transform 'imagenet' \
        --lr 0.1 \
        --eval_funcs 'v1' 'v2' \
        --split $split \
        > ${SAVE_DIR}logfile_${EXP_NUM}.out
done

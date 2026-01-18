#!/bin/bash

config="configs/score_isaid_instance.yaml"
gpus="1"
work_dir="score_isaid_instance"
out_dir="score_isaid_instance"
 

# shift 3
opts="$@"
 
ckpts=("final")
 
for ckpt in ${ckpts[@]}; do

     # NWPU
    python train_net.py --config $config \
    --num-gpus $gpus \
    --dist-url "auto" \
    --eval-only \
    OUTPUT_DIR "$out_dir/eval_${ckpt}/nwpu" \
    DATASETS.TEST "('nwpu_val_instance',)" \
    MODEL.WEIGHTS "$work_dir/model_${ckpt}.pth" \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_ALPHA 0.4 \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_BETA 0.4 \
    $opts
    
    # sota
    python train_net.py --config $config \
    --num-gpus $gpus \
    --dist-url "auto" \
    --eval-only \
    OUTPUT_DIR "$out_dir/eval_${ckpt}/sota" \
    DATASETS.TEST "('sota_val_instance',)" \
    MODEL.WEIGHTS "$work_dir/model_${ckpt}.pth" \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_ALPHA 0.0 \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_BETA 0.4 \
    $opts
 
    # fast
    python train_net.py --config $config \
    --num-gpus $gpus \
    --dist-url "auto" \
    --eval-only \
    OUTPUT_DIR "$out_dir/eval_${ckpt}/fast" \
    DATASETS.TEST "('fast_val_instance',)" \
    MODEL.WEIGHTS "$work_dir/model_${ckpt}.pth" \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_ALPHA 0.2 \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_BETA 0.2 \
    $opts
    
    # sior
    python train_net.py --config $config \
    --num-gpus $gpus \
    --dist-url "auto" \
    --eval-only \
    OUTPUT_DIR "$out_dir/eval_${ckpt}/sior" \
    DATASETS.TEST "('sior_val_instance',)" \
    MODEL.WEIGHTS "$work_dir/model_${ckpt}.pth" \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_ALPHA 0.4 \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_BETA 0.4 \
    $opts

done
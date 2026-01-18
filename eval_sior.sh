#!/bin/bash
 
config="configs/score_sior_instance.yaml"
gpus="1"
work_dir="score_sior_instance"
out_dir="score_sior_instance"
 
# shift 3
opts="$@"
 
ckpts=("final")
 
for ckpt in ${ckpts[@]}; do
 
    # NWPU
    python train_net.py --config $config \
    --num-gpus $gpus \
    --dist-url "auto" \
    --eval-only \
    OUTPUT_DIR "$work_dir/eval_${ckpt}/nwpu" \
    DATASETS.TEST "('nwpu_val_instance',)" \
    MODEL.WEIGHTS "$work_dir/model_${ckpt}.pth" \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_ALPHA 0.2 \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_BETA 0.6 \
    $opts
 
    # sota
    python train_net.py --config $config \
    --num-gpus $gpus \
    --dist-url "auto" \
    --eval-only \
    OUTPUT_DIR "$work_dir/eval_${ckpt}/sota" \
    DATASETS.TEST "('sota_val_instance',)" \
    MODEL.WEIGHTS "$work_dir/model_${ckpt}.pth" \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_ALPHA 0.0 \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_BETA 0.2 \
    $opts

    # fast
    python train_net.py --config $config \
    --num-gpus $gpus \
    --dist-url "auto" \
    --eval-only \
    OUTPUT_DIR "$work_dir/eval_${ckpt}/fast" \
    DATASETS.TEST "('fast_val_instance',)" \
    MODEL.WEIGHTS "$work_dir/model_${ckpt}.pth" \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_ALPHA 0.2 \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_BETA 0.4 \
    $opts

    # isaid
    python train_net.py --config $config \
    --num-gpus $gpus \
    --dist-url "auto" \
    --eval-only \
    OUTPUT_DIR "$work_dir/eval_${ckpt}/isaid" \
    DATASETS.TEST "('isaid_val_instance',)" \
    MODEL.WEIGHTS "$work_dir/model_${ckpt}.pth" \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_ALPHA 0.2 \
    MODEL.SCORE.GEOMETRIC_ENSEMBLE_BETA 0.2 \
    $opts

done
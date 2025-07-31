DATA="/home/sk138/clip_ast/data"
TRAINER=CLIP_AST

DATASET=$1
CFG=clip_ast_few_shot
SEED=2
# SHOTS=$2
# LMBD=1.0

for SHOTS in 1 2 4 8 16
do
    for LR in 5e-8 1e-7 5e-7 1e-6 5e-6
    do
        for LMBD in 0.0 0.1 0.5 1.0 2.5 5.0 7.5 10.0 25.0 50.0 100.0
        do
            DIR=output/${DATASET}/${TRAINER}/${CFG}_seed${SEED}_${SHOTS}shots_lr${LR}_SCL${LMBD}
            # if [ -d "$DIR" ]; then
            #     echo " The results exist at ${DIR}"
            # else
            echo "Run this job and save the output to ${DIR}"
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            OPTIM.LR ${LR} \
            LMBD ${LMBD}
            # fi
        done
    done
done
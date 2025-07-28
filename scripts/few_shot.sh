DATA="/home/sk138/clip_ast/data"
TRAINER=CLIP_AST

DATASET=$1
CFG=clip_ast_few_shot
SHOTS=$2

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo " The results exist at ${DIR}"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --device cuda:0 \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
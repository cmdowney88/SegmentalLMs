#!/bin/sh
source activate mslm

INPUT_FILE=$1
MODEL_NAME=$2
CONFIG_NAME=$3
TRAINING_WORDS=$4

python3 -u -m mslm \
    --input_file=$INPUT_FILE \
    --model_path=Models/${MODEL_NAME} \
    --mode=eval \
    --config_file=Configs/${CONFIG_NAME}.json > Output/${MODEL_NAME}.out

perl Tools/score.pl $TRAINING_WORDS $INPUT_FILE Output/${MODEL_NAME}.out > Output/${MODEL_NAME}.score
tail -n 14 Output/${MODEL_NAME}.score > Output/${MODEL_NAME}.tmp
cat Output/${MODEL_NAME}.tmp > Output/${MODEL_NAME}.score
rm -f Output/${MODEL_NAME}.tmp

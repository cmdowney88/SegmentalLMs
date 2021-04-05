#!/bin/sh
source activate mslm

INPUT_FILE=$1
MODEL_NAME=$2
CONFIG_NAME=$3

python3 -u -m mslm \
    --input_file=$INPUT_FILE \
    --model_path=Models/${MODEL_NAME} \
    --mode=eval \
    --config_file=Configs/${CONFIG_NAME}.json 2> Output/${MODEL_NAME}.bpc

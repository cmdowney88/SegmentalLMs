#!/bin/sh
source activate mslm

MODEL_NAME=$1
TRAIN_FILE=$2
DEV_FILE=$3

python3 -u -m mslm \
    --input_file=$TRAIN_FILE \
    --model_path=Models/${MODEL_NAME} \
    --mode=train \
    --dev_file=$DEV_FILE \
    --config_file=Configs/${MODEL_NAME}.json

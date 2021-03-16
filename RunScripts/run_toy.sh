#!/bin/sh
source activate cp

python3 -u mslm/run_mslm.py \
  Data/toy.txt \
  Models/toy \
  train \
  --dev_file=Data/toy.txt \
  --config_file=Configs/toy_config.json

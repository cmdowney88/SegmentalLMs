#!/bin/sh
source activate mslm

python3 -mu mslm \
  Data/toy.txt \
  Models/toy \
  train \
  --dev_file=Data/toy.txt \
  --config_file=Configs/toy_config.json

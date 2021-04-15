# Segmental Language Models

This repository contains the code from
 _A Masked Segmental Language Model for Natural Language Segmentation_ (2021,
 C.M. Downey, Fei Xia, and Shane Steinert-Threlkeld). The _mslm_ package can be
 used to train and use Segmental Language Models, as first described by Sun and
 Deng (2018) and Kawakami, Dyer, and Blunsom (2019), with either the original
 LSTM-based encoder (a Recurrent SLM), or the span-masking Transformer encoder
 introduced in the present paper

This repository also contains several standard datasets for word-segmentation
 experimentation, as well as utilities and the SIGHAN scoring script (Emerson
 2005)

## Paper Results

The results from the accompanying paper can be found in the `Output` directory.
 `*.csv` files include statistics from the training run, `*.out` contain the
  model output for the entire corpus, `*.score` contain the segmentation scores
  of the model output, and `*.bpc` contain the bits-per-character scores for the
  models on the test sets

## Usage

The top-level scripts for training and experimentation can be found in
 `RunScripts`. Almost all functionality is run through the `__main__.py` script
 in the `mslm` package, which can either train or evaluate/use a model. The
 PyTorch modules for building SLMs can be found in `mslm.segmental_lm`, modules
 for the span-masking Transformer are in `mslm.segmental_transformer`, and
 modules for sequence lattice-based computations are in `mslm.lattice`. The main
 script takes in a configuration object to set most parameters for model
 training and use (see `mslm.mslm_config`). For information on the arguments to
 the main script:
     
    python -m mslm --help

### Environment setup
    pip install -r requirements.txt

This code requires Python >= 3.5

### Training
    ./RunScripts/run_mslm.sh <MODEL/CONFIG_NAME> <TRAIN_FILE> <VALIDATION_FILE>
or 

    python -m mslm --input_file <TRAIN_FILE> \
        --model_path <MODEL_PATH> \
        --mode train \
        --config_file <CONFIG_FILE> \
        --dev_file <DEV_FILE> \
        [--preexisting]

### Evaluation
    ./RunScripts/eval_mslm.sh <INPUT_FILE> <MODEL_PATH> <CONFIG_NAME> <TRAINING_WORDS>

Where `<TRAINING_WORDS>` is a text file containing all of the words from the 
training set

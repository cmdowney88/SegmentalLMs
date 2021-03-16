#!/bin/bash

# Take the script to be run with multiple sets of arguments, the file
# the arguments, one set per line, and an output directory to write stdout and
# stderr
SCRIPT=$1
ARGS_FILE=$2
OUTPUT_DIR=$3

# Set the bash delimiter to be newline for the purposes of looping through the
# lines in the argument file
IFS=$'\n'
set -f
for line in $(cat $ARGS_FILE)
do  
    # If the line is not empty, split on whitespace to get the output name. This
    # is making the assumption that the first argument to the passed script is
    # some sort of configuration name. This can be modified if that assumption
    # doesn't hold
    # Run the script with the given arguments, writing stdout and stderr to the
    # output directory
    if [ "$line" != "" ]; then
        IFS=' '
        SPLIT_ARGS=( $line )
        MODEL_NAME=${SPLIT_ARGS[0]}
        ./${SCRIPT} ${line} > ${OUTPUT_DIR}/${MODEL_NAME}.out 2> ${OUTPUT_DIR}/${MODEL_NAME}.err &
    fi
done

# The jobs command returns IDs separated by newlines, so set the delimiter to be
# newlines again
IFS=$'\n'
# Loop through the forked subprocesses, waiting for each one. This should have
# the effect of waiting for each one to finish
JOBS=$(jobs -p)
for job in $JOBS
do
    wait $job
done

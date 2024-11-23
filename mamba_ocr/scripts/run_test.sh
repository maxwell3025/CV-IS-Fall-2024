#!/bin/bash
#
# This script uploads the code to Unity, goes through the full test sequence,
# and finally downloads the results to the local repo.
# 
# This script is designed to be run on the client.
#
# Optionally, you can pass in --sync in order to wait for the script to complete.

cd $(dirname $0)/..
echo "Local working directory is $(pwd)"

test -e .env || { echo "Error: Failed to locate .env" ; exit 1 ; }
source .env

MODULE_NAME=$(basename $(pwd))

test $UNITY_WORK_HOME || { echo "UNITY_WORK_HOME not defined" ; exit 1 ; }
echo "Remote working directory is $UNITY_WORK_HOME/$MODULE_NAME"

./scripts/push.sh

if [[ "$1" == "--sync" ]]; then
    echo "================================================================================"
    ssh -t unity "cd $UNITY_WORK_HOME/mamba_ocr ; . ./slurm_constraints ; srun ./scripts/unity_run_test.sh"
    echo "================================================================================"
    ./scripts/pull.sh
else
    ssh -t unity "cd $UNITY_WORK_HOME/mamba_ocr ; . ./slurm_constraints ; sbatch ./scripts/unity_run_test.sh"
    # echo "DEPRECATED"
fi

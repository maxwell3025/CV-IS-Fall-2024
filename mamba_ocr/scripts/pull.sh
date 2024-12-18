#!/bin/bash
#
# This pulls models and logs from Unity.

cd $(dirname $0)/..

source .env

MODULE_NAME=$(basename $(pwd))

rsync -rt unity:$UNITY_WORK_HOME/$MODULE_NAME/latest.log .

rsync -rt unity:$UNITY_WORK_HOME/$MODULE_NAME/output/ .output

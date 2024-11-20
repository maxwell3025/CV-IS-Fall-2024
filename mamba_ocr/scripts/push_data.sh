#!/bin/bash
#
# Push local code and configs to unity

cd $(dirname $0)/..
source .env

MODULE_NAME=$(basename $(pwd))

rsync \
    -Rr \
    data \
    unity:$UNITY_WORK_HOME/$MODULE_NAME/data

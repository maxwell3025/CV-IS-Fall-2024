#!/bin/bash
#
# Push local code and configs to unity

cd $(dirname $0)/..
source .env

MODULE_NAME=$(basename $(pwd))

rsync \
    -R \
    .env \
    $(git ls-files --others --exclude-standard --cached) \
    unity:$UNITY_WORK_HOME/$MODULE_NAME
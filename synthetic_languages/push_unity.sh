#!/bin/bash
source .env
FOLDER_BASENAME=$(basename $(dirname $(realpath $0)))
rsync \
    -R \
    $(git ls-files --others --exclude-standard --cached) \
    unity:$UNITY_WORK_BASE/$FOLDER_BASENAME \
    --ignore-missing-args

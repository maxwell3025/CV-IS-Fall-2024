#!/bin/bash

source .env
SCRIPT_DIR=$(dirname $(realpath $0))
BASE_DIR=$(realpath "$SCRIPT_DIR/..")
FOLDER_BASENAME=$(basename $BASE_DIR)
rsync \
    -R \
    "$BASE_DIR/.env" \
    $(git ls-files --others --exclude-standard --cached) \
    unity:$UNITY_WORK_BASE/$FOLDER_BASENAME \
    --ignore-missing-args

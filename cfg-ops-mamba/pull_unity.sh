#!/bin/bash
source .env
LOCAL_WORK_FOLDER=$(dirname $(realpath $0))
FOLDER_BASENAME=$(basename $(dirname $(realpath $0)))
rsync -r unity:$UNITY_WORK_BASE/$FOLDER_BASENAME/output/ $LOCAL_WORK_FOLDER/output/

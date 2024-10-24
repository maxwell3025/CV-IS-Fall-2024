source .env
LOCAL_WORK_FOLDER=$(dirname "$(realpath $0)")

rsync -r unity:$UNITY_WORK_FOLDER/output/ $LOCAL_WORK_FOLDER/output/

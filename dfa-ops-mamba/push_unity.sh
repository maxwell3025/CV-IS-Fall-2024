source .env
rsync -R .env $(git ls-files --others --exclude-standard --cached) unity:$UNITY_WORK_FOLDER --ignore-missing-args

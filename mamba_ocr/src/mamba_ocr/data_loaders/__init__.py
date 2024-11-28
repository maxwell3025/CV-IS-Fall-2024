from . import mscoco_task
from . import iam_task
from . import latex_task
from . import ocr_task
from . import ocr_task_base
from . import partition_task

datasets = {
    "mscoco": mscoco_task.MsCocoTask,
    "iam": iam_task.IamTask,
    "latex": latex_task.LatexTask,
}
from . import sequence_stack
from . import medmamba_stack
from . import ocr_model

models = {
    "sequence_stack": sequence_stack.SequenceStack,
    "medmamba_stack": medmamba_stack.MedmambaStack,
}
from . import ocr_task_base

class PartitionTask(ocr_task_base.OcrTaskBase):
    def __init__(
        self,
        parent: ocr_task_base.OcrTaskBase,
        split: tuple[float, float]
    ):
        super().__init__()
        start_index = round(len(parent.contexts) * split[0])
        end_index = round(len(parent.contexts) * split[1])
        self._contexts = parent.contexts[start_index: end_index]
        self.parent = parent

    @property
    def contexts(self):
        return self._contexts
        
    @property
    def d_alphabet(self) -> int:
        return self.parent.d_alphabet

    @property
    def d_color(self) -> int:
        return 3

    @property
    def d_positional_encoding(self) -> int:
        return self.parent.d_positional_encoding

    def get_alphabet_index(self, char: str) -> int:
        return self.parent.get_alphabet_index(char)
        
    def get_index_alphabet(self, index: int) -> str:
        return self.parent.get_index_alphabet(index)
        
from . import ocr_task
import logging
import numpy
import torch
from abc import abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

class OcrTaskBase(ocr_task.OcrTask):
    def __init__(
        self,
    ):
        self.current_index = (0, 0)
    
    @property
    @abstractmethod
    def contexts(self) -> list[tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Get the list of contexts associated with this dataset.
        """
        pass

    def get_batch(
        self,
        batch_size: int,
    ) -> list[tuple[list[torch.Tensor], list[torch.Tensor]]]:
        context_id, instance_id = self.current_index
        if instance_id != 0:
            context_id += 1
            instance_id = 0
        if context_id + batch_size >= len(self.contexts):
            context_id = 0
        batch = self.contexts[context_id:context_id + batch_size]
        context_id += batch_size
        self.current_index = context_id, instance_id
        return batch
        
    def get_batch_no_context(
        self,
        batch_size: int,
    ) -> list[tuple[list[torch.Tensor], list[torch.Tensor]]]:
        context_id, instance_id = self.current_index
        batch = []
        for i in range(batch_size):
            current_context = self.contexts[context_id]
            features, labels = current_context
            feature = features[instance_id]
            label = labels[instance_id]
            batch.append(([feature], [label]))
            instance_id += 1
            if instance_id >= len(features):
                instance_id = 0
                context_id += 1
            if context_id >= len(self.contexts):
                context_id = 0
        self.current_index = context_id, instance_id
        return batch

    def batches(self, batch_size: int):
        context_id = 0
        while context_id <= len(self.contexts) - batch_size:
            yield self.contexts[context_id:context_id + batch_size]
            context_id += batch_size

    def batches_no_context(self, batch_size: int):
        context_id, instance_id = 0, 0

        while context_id < len(self.contexts):
            batch = []
            for i in range(batch_size):
                current_context = self.contexts[context_id]
                features, labels = current_context
                feature = features[instance_id]
                label = labels[instance_id]
                batch.append(([feature], [label]))
                instance_id += 1
                while context_id < len(self.contexts) and instance_id >= len(self.contexts[context_id][0]):
                    instance_id = 0
                    context_id += 1
                if context_id >= len(self.contexts):
                    break
            yield batch
from . import ocr_task
import json
import numpy
import torch
from torch.nn import functional as torch_functional
from torchvision.transforms import functional as torchvision_functional
from PIL import Image

class MsCocoText:
    def __init__(self, data_object: dict[str, str]):
        self.anns: dict[str, any] = data_object["anns"]
        self.imgs: dict[str, any] = data_object["imgs"]
        self.img_to_anns: dict[str, any] = data_object["imgToAnns"]
        
positional_encoding_vectors: list[tuple[float, float]] = [
    (1, 0),
    (0, 1),
    (1, 1),
]

class MsCocoTask(ocr_task.OcrTask):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        with open(f"{data_path}/cocotext.v2.json") as annotation_file:
            data: dict[any, any] = json.load(annotation_file)
            dataset = MsCocoText(data)

        # Here, we populate alphabet_set, which is a map from characters to IDs
        alphabet_set: set[str] = set()
        for annotation in dataset.anns.values():
            alphabet_set.update(list(annotation["utf8_string"]))
        self.alphabet: dict[str, int] = dict((char, index) for (index, char) in
                                             enumerate(alphabet_set))
        self.d_alphabet = len(alphabet_set)

        positional_encoding_size = len(positional_encoding_vectors) * 2
        token_length = len(alphabet_set) + positional_encoding_size + 3
        self.sequences: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []
        for image_id, annotation_ids in dataset.img_to_anns.items():
            if len(annotation_ids) == 0:
                continue
            file_name = dataset.imgs[image_id]["file_name"]
            with Image.open(f"{data_path}/train2014/{file_name}") as image:
                current_sequence = []
                current_mask = []
                for annotation_id in annotation_ids:
                    annotation = dataset.anns[str(annotation_id)]
                    if(len(annotation["utf8_string"]) == 0):
                        continue

                    x, y, w, h = annotation["bbox"]
                    x, y, w, h = round(x), round(y), round(w), round(h)
                    selection = image.crop((x, y, x+w, y+h))
                    image_tensor = torchvision_functional.pil_to_tensor(selection)/255
                    assert image_tensor.shape == (3, h, w)

                    x_tensor = numpy.arange(w) + x
                    x_tensor = x_tensor[numpy.newaxis, :]
                    x_tensor = numpy.broadcast_to(x_tensor, (h, w))
                    assert x_tensor.shape == (h, w)

                    y_tensor = numpy.arange(h) + y
                    y_tensor = y_tensor[:, numpy.newaxis]
                    y_tensor = numpy.broadcast_to(y_tensor, (h, w))
                    assert y_tensor.shape == (h, w)

                    positional_encoding: list[numpy.ndarray] = []
                    for x_dependence, y_dependence in positional_encoding_vectors:
                        dot_product = x_tensor * x_dependence + y_tensor * y_dependence
                        positional_encoding.append(numpy.sin(dot_product))
                        positional_encoding.append(numpy.cos(dot_product))
                    positional_encoding: numpy.ndarray = numpy.stack(positional_encoding, axis=0)
                    assert positional_encoding.shape == (positional_encoding_size, h, w)
                    positional_encoding: torch.Tensor = torch.from_numpy(positional_encoding)

                    channel_padding = torch.zeros((len(alphabet_set), h, w))

                    full_image = torch.cat((image_tensor, positional_encoding, channel_padding), dim=0)
                    full_image = full_image.transpose(1, 2)
                    sequence_image = full_image.flatten(1, 2)
                    sequence_image = sequence_image.transpose(0, 1)
                    assert sequence_image.shape == (w * h, token_length)

                    text: str = annotation["utf8_string"]
                    text_length = len(text)

                    text: list[str] = list(text)
                    text: list[int] = [self.alphabet[char] for char in text]
                    text: torch.Tensor = torch.tensor(text)
                    text = torch_functional.one_hot(text, num_classes=len(alphabet_set))
                    text = torch.cat((torch.zeros((text_length, positional_encoding_size + 3)), text), dim=1)
                    assert text.shape == (len(annotation["utf8_string"]), token_length)

                    current_sequence.append(sequence_image)
                    current_sequence.append(text)

                    current_mask.append(torch.zeros(sequence_image.shape[0]))
                    current_mask.append(torch.ones(text.shape[0]))
                if len(current_sequence) != 0:
                    self.sequences.append(torch.cat(current_sequence))
                    self.masks.append(torch.cat(current_mask))
    
    def get_d_color(self):
        return 3
    def get_d_positional_encoding(self) -> int:
        return len(positional_encoding_vectors) * 2
    def get_d_alphabet(self) -> int:
        return self.d_alphabet

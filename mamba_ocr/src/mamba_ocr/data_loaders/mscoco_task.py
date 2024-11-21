from . import ocr_task_base
import json
import numpy
import torch
from torch.nn import functional as torch_functional
from torchvision.transforms import functional as torchvision_functional
from PIL import Image

class MsCocoAnnotation:
    def __init__(self, data_object: dict[str, str]):
        self.mask: list[float] = data_object["mask"]
        self.clazz: str = data_object["class"]
        self.bbox: list[float] = data_object["bbox"]
        self.image_id: int = data_object["image_id"]
        self.id: int = data_object["id"]
        self.language: str = data_object["language"]
        self.area: float = data_object["area"]
        self.utf8_string: str = data_object["utf8_string"]
        self.legibility: str = data_object["legibility"]
        
class MsCocoText:
    def __init__(self, data_object: dict[str, str]):
        self.anns: dict[str, MsCocoAnnotation] = dict(
            (key, MsCocoAnnotation(value)) for (key, value) in
            data_object["anns"].items()
        )
        self.imgs: dict[str, any] = data_object["imgs"]
        self.img_to_anns: dict[str, any] = data_object["imgToAnns"]

class MsCocoTask(ocr_task_base.OcrTaskBase):
    def __init__(
        self,
        data_path: str,
        positional_encoding_vectors: numpy.ndarray,
        train_test_val_split: tuple[float, float],
    ) -> None:
        """Initializes an MsCocoTask instance.

        Args:
            data_path: A string referencing the base folder for the dataset.
            positional_encoding_vectors: A numpy array with the shape [n, 2],
                where n is the number of encoding vectors.
        """
        with open(f"{data_path}/cocotext.v2.json") as annotation_file:
            data: dict[any, any] = json.load(annotation_file)
            dataset = MsCocoText(data)

        # Here, we populate alphabet_set, which is a map from characters to IDs
        alphabet_set: set[str] = set()
        for annotation in dataset.anns.values():
            alphabet_set.update(list(annotation.utf8_string))
        self.alphabet: dict[str, int] = dict(enumerate(alphabet_set))
        self.reverse_alphabet: dict[str, int] = dict((char, index) for
                                                     (index, char) in
                                                     self.alphabet.items())

        self.positional_encoding_vectors = positional_encoding_vectors
        features: list[list[torch.Tensor]] = []
        masks: list[list[torch.Tensor]] = []
        for image_id, annotation_ids in dataset.img_to_anns.items():
            if len(annotation_ids) == 0:
                continue

            file_name = dataset.imgs[image_id]["file_name"]
            with Image.open(f"{data_path}/train2014/{file_name}") as image:
                context_features = []
                context_masks = []
                for annotation_id in annotation_ids:
                    annotation = dataset.anns[str(annotation_id)]
                    conversion_result = self.convert_to_sequence(image, annotation)
                    if conversion_result == None:
                        continue
                    feature, mask = conversion_result
                    context_features.append(feature)
                    context_masks.append(mask)

                if len(context_features) != 0:
                    features.append(context_features)
                    masks.append(context_masks)
            break
        super(MsCocoTask, self).__init__(features, masks, train_test_val_split)

    def convert_to_sequence(
        self,
        image: Image.Image,
        annotation: MsCocoAnnotation,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if(len(annotation.utf8_string) == 0):
            return None

        x, y, w, h = annotation.bbox
        x, y, w, h = round(x), round(y), round(w), round(h)
        selection = image.crop((x, y, x+w, y+h))
        image_tensor = torchvision_functional.pil_to_tensor(selection)/255
        assert image_tensor.shape == (3, h, w)

        positional_encoding: torch.Tensor = self.get_position_tensor(
            x, y, w, h
        )
        assert positional_encoding.shape == (self.get_d_positional_encoding(), h, w)

        channel_padding = torch.zeros((self.get_d_alphabet(), h, w))
        assert channel_padding.shape == (self.get_d_alphabet(), h, w)

        feature_image = torch.cat((
            image_tensor,
            positional_encoding,
            channel_padding
        ), dim=0)
        assert feature_image.shape == (self.get_d_input(), h, w)
        feature_image = feature_image.transpose(1, 2)
        assert feature_image.shape == (self.get_d_input(), w, h)
        feature_sequence_image = feature_image.flatten(1, 2)
        assert feature_sequence_image.shape == (self.get_d_input(), w * h)
        feature_sequence_image = feature_sequence_image.transpose(0, 1)
        assert feature_sequence_image.shape == (w * h, self.get_d_input())

        label: str = annotation.utf8_string
        label_length = len(label)

        label: list[str] = list(label)
        label: list[int] = [self.reverse_alphabet[char] for char in label]
        label: torch.Tensor = torch.tensor(label)
        label = torch_functional.one_hot(label, num_classes=self.get_d_alphabet())
        assert label.shape == (label_length, self.get_d_alphabet())
        channel_padding = torch.zeros((label_length, self.get_d_input() - self.get_d_alphabet()))

        label = torch.cat((channel_padding, label), dim=1)
        assert label.shape == (label_length, self.get_d_input())

        feature_sequence = torch.cat((feature_sequence_image, label), dim=0)

        mask_sequence = torch.cat((
            torch.zeros(feature_sequence_image.shape[0]),
            torch.ones(label.shape[0])
        ), dim=0)

        return feature_sequence, mask_sequence
    
    def get_d_color(self):
        return 3

    def get_d_positional_encoding(self) -> int:
        return self.positional_encoding_vectors.size

    def get_d_alphabet(self) -> int:
        return len(self.alphabet)

    def get_d_input(self) -> int:
        return self.get_d_color() + self.get_d_alphabet() + self.get_d_positional_encoding()

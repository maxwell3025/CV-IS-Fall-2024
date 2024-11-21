from . import ocr_task_base
import io
import numpy
import pandas
from PIL import Image
import torch
from torch.nn import functional as torch_functional
from torchvision.transforms import functional as torchvision_functional

class IamTask(ocr_task_base.OcrTaskBase):
    def __init__(
        self,
        positional_encoding_vectors: numpy.ndarray,
        train_test_val_split: tuple[float, float],
    ) -> None:
        super(IamTask, self).__init__([], [], (0, 0))
        print("Loading IAM dataset...")
        splits = dict(
            train="data/train-00000-of-00001-bfc7b63751c36ab0.parquet",
            test="data/test-00000-of-00001-4cae70e6a03872e2.parquet",
            val="data/val-00000-of-00001-472467affea948eb.parquet",
        )
        df1 = pandas.read_parquet("hf://datasets/priyank-m/IAM_words_text_recognition/" + splits["train"])
        df2 = pandas.read_parquet("hf://datasets/priyank-m/IAM_words_text_recognition/" + splits["test"])
        df3 = pandas.read_parquet("hf://datasets/priyank-m/IAM_words_text_recognition/" + splits["val"])
        df = pandas.concat([df1, df2, df3])
        print("Finished loading IAM dataset")

        print("Generating alphabet...")
        alphabet_set: set[str] = set()

        for row in df.itertuples():
            alphabet_set.update(*list(row.text))

        self.alphabet = dict(enumerate(alphabet_set))
        self.reverse_alphabet = dict((value, key) for (key, value) in self.alphabet.items())
        print("Finished generating alphabet")

        print("Serializing dataset...")
        self.positional_encoding_vectors = positional_encoding_vectors
        features: list[list[torch.Tensor]] = []
        masks: list[list[torch.Tensor]] = []
        for row in df.itertuples():
            image = Image.open(io.BytesIO(row.image["bytes"]))
            text = row.text
            feature, mask = self.convert_to_sequence(image, text)
            features.append([feature])
            masks.append([mask])
            break
        print("Finished serializing dataset")
        super(IamTask, self).__init__(features, masks, train_test_val_split)

    def convert_to_sequence(
        self,
        image: Image.Image,
        text: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w, h = image.width, image.height
        image_tensor = torchvision_functional.pil_to_tensor(image)/255
        assert image_tensor.shape == (self.get_d_color(), h, w)

        positional_encoding: torch.Tensor = self.get_position_tensor(
            0, 0, w, h
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

        label: str = text
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
        return 1

    def get_d_positional_encoding(self) -> int:
        return self.positional_encoding_vectors.size

    def get_d_alphabet(self) -> int:
        return len(self.alphabet)

    def get_d_input(self) -> int:
        return self.get_d_color() + self.get_d_alphabet() + self.get_d_positional_encoding()

from . import ocr_task_base
from . import util
import itertools
import json
import logging
import math
import numpy
import torch
from torchvision.transforms import functional as torchvision_functional
from typing import Any
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

class MsCocoAnnotation:
    def __init__(self, data_object: dict[str, Any]):
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
    def __init__(self, data_object: dict[str, Any]):
        self.anns: dict[str, MsCocoAnnotation] = dict(
            (key, MsCocoAnnotation(value)) for (key, value) in
            data_object["anns"].items()
        )
        self.imgs: dict[str, Any] = data_object["imgs"]
        self.img_to_anns: dict[str, list[int]] = data_object["imgToAnns"]

class MsCocoTask(ocr_task_base.OcrTaskBase):
    """A dataset containing the MSCOCO-Text dataset, with cropped word images,
    positional encodings, and one-hot encoded text.
    
    Attributes:
        default_height: The height that all word images are scaled to.
        positional_encoding_vectors: A Numpy array containing all of the
            positional encoding vectors.
        encode_relative_position_norm: Whether the position within the word
            image should be encoded.
        encode_absolute_position_norm: Whether the position within the image
            should be encoded.
        encode_relative_position_px: Whether the pixel coord within the word
            should be encoded.
        encode_absolute_position_px: Whether the pixel coord within the image
            should be encoded.
    """
    default_height: int
    positional_encoding_vectors: numpy.ndarray
    encode_relative_position_norm: bool
    encode_absolute_position_norm: bool
    encode_relative_position_px: bool
    encode_absolute_position_px: bool

    def __init__(
        self,
        data_path: str,
        positional_encoding_vectors: list[list[float]],
        default_height: int,
        encode_relative_position_norm: bool = True,
        encode_absolute_position_norm: bool = True,
        encode_relative_position_px: bool = True,
        encode_absolute_position_px: bool = True,
    ) -> None:
        self.default_height = default_height
        self.positional_encoding_vectors = numpy.array(positional_encoding_vectors)
        self.encode_relative_position_norm = encode_relative_position_norm
        self.encode_absolute_position_norm = encode_absolute_position_norm
        self.encode_relative_position_px   = encode_relative_position_px
        self.encode_absolute_position_px   = encode_absolute_position_px

        logger.info("Loading annotation file")
        with open(f"{data_path}/cocotext.v2.json") as annotation_file:
            metadata = json.load(annotation_file)
            metadata = MsCocoText(metadata)

        logger.info("Generating alphabet")
        # We populate `alphabet` and `reverse_alphabet` by collecting the set of
        # all characters in the metadata object and enumerating them.
        alphabet_map: dict[str, int] = {}
        for annotation in metadata.anns.values():
            for char in list(annotation.utf8_string):
                if alphabet_map.get(char, None) == None:
                    alphabet_map[char] = 1
                else:
                    alphabet_map[char] += 1

        common_letters = set(
            char for char in alphabet_map.keys() if alphabet_map[char] >= 100
        )

        rare_letters = set(
            char for char in alphabet_map.keys() if alphabet_map[char] < 100
        )

        #region entropy calculation
        total_chars = sum(alphabet_map.values())
        entropy = sum(
            -alphabet_map[char] / total_chars * math.log(
                alphabet_map[char] / total_chars
            )
            for char in common_letters
        )
        total_rare = sum(alphabet_map[char] for char in rare_letters)
        entropy -= total_rare / total_chars * math.log(
            total_rare / total_chars
        )
        logger.info(f"Character-wise entropy of mscoco: {entropy}")
        #endregion

        self.alphabet = dict(enumerate(common_letters))

        self.reverse_alphabet = dict(
            (char, index) for (index, char) in self.alphabet.items()
        )

        self.reverse_alphabet.update((char, len(common_letters)) for char in rare_letters)

        logger.info("Generating dataset")
        # We populate `_contexts` by iterating through the annotations for each
        # image.
        self._contexts = []
        for image_id, annotation_ids in itertools.islice(metadata.img_to_anns.items(), 6400):
            if len(annotation_ids) == 0:
                continue
            file_name = metadata.imgs[image_id]["file_name"]
            current_context = ([], [])
            total_image_size = 0
            MAX_IMAGE_SIZE = 64*64*100

            with Image.open(f"{data_path}/train2014/{file_name}") as context_image:
                for annotation_id in annotation_ids:
                    annotation = metadata.anns[str(annotation_id)]

                    label = list(annotation.utf8_string)
                    if len(label) <= 1:
                        continue
                    label = map(lambda x: self.get_alphabet_index(x), label)
                    label = list(label) + [self.d_alphabet - 1]
                    label = torch.tensor(label)
                    label = torch.nn.functional.one_hot(label)

                    x, y, w, h = annotation.bbox
                    x_min = x
                    x_max = x + w
                    y_min = y
                    y_max = y + h
                    feature = context_image.crop((x_min, y_min, x_max, y_max))
                    rescale_factor = self.default_height / h
                    w_new = round(w * rescale_factor)
                    h_new = round(h * rescale_factor)
                    feature = feature.resize((w_new, h_new))
                    feature = torchvision_functional.pil_to_tensor(feature)
                    feature = feature/255
                    assert feature.shape[1] == h_new
                    assert feature.shape[2] == w_new
                    feature = feature.broadcast_to((3, h_new, w_new))
                    assert feature.shape == (3, h_new, w_new)
                    feature_stack = [feature]

                    # We optionally append positional encodings as extra
                    # channels in the image.
                    if encode_relative_position_norm:
                        feature_stack.append(util.get_position_tensor(
                            w=w_new,
                            h=h_new,
                            x_min=0,
                            y_min=0,
                            x_max=1,
                            y_max=1,
                            positional_encoding_vectors=self.positional_encoding_vectors
                        ))

                    if encode_absolute_position_norm:
                        feature_stack.append(util.get_position_tensor(
                            w=w_new,
                            h=h_new,
                            x_min=x_min / context_image.width,
                            y_min=y_min / context_image.height,
                            x_max=x_max / context_image.width,
                            y_max=y_max / context_image.height,
                            positional_encoding_vectors=self.positional_encoding_vectors
                        ))

                    if encode_relative_position_px:
                        feature_stack.append(util.get_position_tensor(
                            w=w_new,
                            h=h_new,
                            x_min=0,
                            y_min=0,
                            x_max=w,
                            y_max=h,
                            positional_encoding_vectors=self.positional_encoding_vectors
                        ))

                    if encode_absolute_position_px:
                        feature_stack.append(util.get_position_tensor(
                            w=w_new,
                            h=h_new,
                            x_min=x_min,
                            y_min=y_min,
                            x_max=x_max,
                            y_max=y_max,
                            positional_encoding_vectors=self.positional_encoding_vectors
                        ))

                    feature = torch.cat(feature_stack)

                    total_image_size += feature.numel() // feature.shape[0]
                    if total_image_size > MAX_IMAGE_SIZE:
                        break

                    current_context[0].append(feature.cpu())
                    current_context[1].append(label.cpu())

            if len(current_context[0]) != 0:
                self._contexts.append(current_context)
        logger.info("Finished loading mscoco")

    @property
    def contexts(self):
        return self._contexts
        
    @property
    def d_alphabet(self) -> int:
        return len(self.alphabet) + 1

    @property
    def d_color(self) -> int:
        return 3

    @property
    def d_positional_encoding(self) -> int:
        return self.positional_encoding_vectors.shape[0] * (
            self.encode_absolute_position_norm +
            self.encode_absolute_position_px +
            self.encode_relative_position_norm +
            self.encode_relative_position_px
        )

    def get_alphabet_index(self, char: str) -> int:
        return self.reverse_alphabet[char]
        
    def get_index_alphabet(self, index: int) -> str:
        return self.alphabet[index]

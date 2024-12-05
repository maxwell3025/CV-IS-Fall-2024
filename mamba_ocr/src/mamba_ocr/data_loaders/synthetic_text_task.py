from . import ocr_task_base
from . import util
import english_words
import numpy
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import random
import torch
from torchvision.transforms import functional as torchvision_functional


class SyntheticTextTask(ocr_task_base.OcrTaskBase):
    def __init__(
        self,
        n_contexts: int,
        context_size: int,
        positional_encoding_vectors: list[list[float]],
        default_height: int,
        encode_relative_position_norm: bool = True,
        encode_relative_position_px: bool = True,
        single_words: bool = False,
        monochrome: bool = False,
    ):
        self.default_height = default_height
        self.positional_encoding_vectors = numpy.array(positional_encoding_vectors)
        self.encode_relative_position_norm = encode_relative_position_norm
        self.encode_relative_position_px   = encode_relative_position_px

        words = english_words.get_english_words_set(
            sources=['gcide'],
            alpha=True,
            lower=True,
        )
        words = list(words)

        if single_words:
            words = [
                '1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9',
                '0'
            ]
        self._contexts = []
        for context_number in range(n_contexts):
            if monochrome :
                background_color = (
                    random.randint(255, 255), 
                    random.randint(255, 255), 
                    random.randint(255, 255), 
                )
                foreground_color = (
                    random.randint(0, 0), 
                    random.randint(0, 0), 
                    random.randint(0, 0), 
                )
            else:
                background_color = (
                    random.randint(0, 255), 
                    random.randint(0, 255), 
                    random.randint(0, 255), 
                )
                foreground_color = (
                    random.randint(0, 255), 
                    random.randint(0, 255), 
                    random.randint(0, 255), 
                )

            # font = ImageFont.truetype(
            #     # "Pillow/Tests/fonts/ArefRuqaa-Regular.ttf",
            #     size=default_height,
            # )
            font = ImageFont.load_default(
                size=default_height * 0.75,
            )
            current_context = ([], [])
            for instance_index in range(context_size):
                label = random.choice(words)
                while len(label) < 10 and len(label) > 2:
                    label = random.choice(words)
                feature = Image.new(
                    mode="RGB",
                    size=(
                        round(default_height * len(label) * 0.5),
                        round(default_height),
                    ),
                    color=background_color, # type: ignore
                )
                canvas = ImageDraw.Draw(feature)
                canvas.text(
                    xy=(0, 0),
                    text=label,
                    fill=foreground_color,
                    font=font,
                )
                
                label = list(label)
                label = map(lambda x: self.get_alphabet_index(x), label)
                label = list(label) + [self.d_alphabet - 1]
                label = torch.tensor(label)
                label = torch.nn.functional.one_hot(label, self.d_alphabet)

                w, h = feature.width, feature.height

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

                feature = torch.cat(feature_stack)

                current_context[0].append(feature.cpu())
                current_context[1].append(label.cpu())
            self._contexts.append(current_context)
        
    def get_alphabet_index(self, char: str) -> int:
        if ord(char) >= ord("a") and ord(char) <= ord("z"):
            return ord(char) - ord("a")
        if ord(char) >= ord("0") and ord(char) <= ord("9"):
            return ord(char) - ord("0") + 26
        raise ValueError(f"\"{char}\" has no index")

    def get_index_alphabet(self, index: int) -> str:
        if index >= 36:
            return ""
        if index >= 26:
            return chr(ord("0") + index - 26)
        return chr(ord("a") + index)

    @property
    def contexts(self):
        return self._contexts

    @property
    def d_alphabet(self):
        return 37

    @property
    def d_color(self) -> int:
        return 3

    @property
    def d_positional_encoding(self) -> int:
        return self.positional_encoding_vectors.shape[0] * 2 * (
            self.encode_relative_position_norm +
            self.encode_relative_position_px
        )


import json
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import List, Tuple
import numpy as np
from ocr_task_base import OcrTaskBase

class OcrDataLoader(OcrTaskBase):
    def __init__(
            self,
            data_root: str,
            image_size: Tuple[int, int] = (224, 224)
    ):
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size
        labels_path = os.path.join(self.data_root, 'data', 'train_labels.txt')
        print(f"Loading labels from: {labels_path}")
        self.image_data = self._load_and_parse_labels(labels_path)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}

    def _load_and_parse_labels(self, labels_path: str) -> List[dict]:
        with open(labels_path, 'r') as f:
            data = json.load(f)
        print(f"Found {len(data['imgs'])} images in labels file")
        processed_data = []
        for img_id, img_info in data['imgs'].items():
            if img_info['set'] == 'train':
                processed_data.append({
                    'image_path': img_info['file_name'],
                    'id': img_info['id'],
                    'width': img_info['width'],
                    'height': img_info['height']
                })
        print(f"Processed {len(processed_data)} training images")
        return processed_data

    @property
    def d_alphabet(self) -> int:
        return len(self.alphabet)

    @property
    def d_color(self) -> int:
        return 3

    @property
    def d_positional_encoding(self) -> int:
        return 16

    def get_alphabet_index(self, char: str) -> int:
        return self.char_to_idx.get(char, -1)

    def get_index_alphabet(self, index: int) -> str:
        if 0 <= index < len(self.alphabet):
            return self.alphabet[index]
        return ''

    def _load_and_process_image(self, image_path: str) -> torch.Tensor:
        corrected_path = image_path.replace('train/', 'train_images/')
        full_path = os.path.join(self.data_root, 'data', corrected_path)
        if not os.path.exists(full_path):
            print(f"Warning: Image not found at {full_path}")
            return None
        image = Image.open(full_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        return transform(image)

    def _create_dummy_label(self, image_id: str) -> torch.Tensor:
        return self._create_one_hot_label(image_id[:8])

    def _create_one_hot_label(self, text: str) -> torch.Tensor:
        indices = [self.get_alphabet_index(c) for c in text]
        valid_indices = [i for i in indices if i != -1]
        one_hot = torch.zeros(len(valid_indices), self.d_alphabet)
        for idx, char_idx in enumerate(valid_indices):
            one_hot[idx][char_idx] = 1
        return one_hot

    @property
    def contexts(self) -> List[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        all_contexts = []
        for img_data in self.image_data:
            image_tensor = self._load_and_process_image(img_data['image_path'])
            if image_tensor is None:
                continue
            label_tensor = self._create_dummy_label(img_data['id'])
            all_contexts.append(([image_tensor], [label_tensor]))
        print(f"Created {len(all_contexts)} contexts")
        return all_contexts
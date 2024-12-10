import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import torch
from typing import List, Tuple
from ocr_data_loader import OcrDataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def test_ocr_data_loader():
    project_root = os.path.abspath(os.path.join(src_dir, '..'))
    print(f"Project root: {project_root}")
    positional_encoding_vectors: np.ndarray = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
    ])
    data_loader = OcrDataLoader(
        data_root=project_root,
        image_size=(224, 224)
    )
    try:
        batch = data_loader.get_batch(batch_size=1)
        print(batch)
        first_context = batch[0]
        first_features = first_context[0][0]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        img_display = first_features.numpy().transpose(1, 2, 0)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        plt.imshow(img_display)
        plt.title('First Image in Batch')
        plt.subplot(1, 2, 2)
        feature_slice = first_features.numpy().transpose()[:, -100:]
        plt.matshow(feature_slice, fignum=False)
        plt.title('Last 100 Columns of Features')
        plt.tight_layout()
        plt.show()
        print(f"Feature shape: {first_features.shape}")
        print(f"Number of images in first context: {len(first_context[0])}")
        print(f"Number of labels in first context: {len(first_context[1])}")
        first_label = first_context[1][0]
        plt.figure(figsize=(10, 2))
        plt.imshow(first_label.numpy(), aspect='auto', cmap='binary')
        plt.title('First Label One-Hot Encoding')
        plt.colorbar()
        plt.show()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ocr_data_loader()
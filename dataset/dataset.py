from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import XGLMTokenizer

from .custom_generator import Generator


class GenDataset(Dataset):
    def __init__(
        self,
        dataset_len: int = -1,
        transform: Optional[Callable] = None,
        generator: Optional[Generator] = None,
        tokenizer: XGLMTokenizer = None,
    ):
        self.transform = transform
        self.generator = generator
        self.tokenizer = tokenizer

        self.dataset_len = len(generator) if dataset_len == -1 else dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        img, label = self.generator.get(index)

        if self.transform is not None:
            img = self.transform(img)
        assert type(label) != str
        input_ids = self.tokenizer(label.text, return_tensors="pt")[
            "input_ids"
        ].squeeze(0)  # dim: (seq_len,)
        # add EOS
        try:
            input_ids = torch.cat(
                (input_ids, torch.tensor([self.tokenizer.eos_token_id]))
            )
        except Exception as e:
            print(f"e: {e}")
            _i = self.tokenizer(label.text, return_tensors="pt")["input_ids"]
            print(f"text: {label.text}\ninput_ids: {input_ids}\ntokened ids: {_i}")
            exit(0)

        color = torch.tensor(label.color, dtype=torch.int)
        stroke_color = torch.tensor(label.stroke_color, dtype=torch.int)

        return img, input_ids, color, stroke_color, self.tokenizer.pad_token_id


def imgcv_from_tensor(img_t: torch.Tensor) -> np.ndarray:
    img_np = img_t.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = (img_np * 255).astype(np.uint8)
    return img_np

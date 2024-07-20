from typing import Dict, List, Tuple
import torch
from torch import Tensor


def collator_gened(batch: List[Tuple[Tensor, Tensor, Tensor, Tensor, int]]) -> Dict:
    img, input_ids, color, stroke_color, pad_ids = zip(*batch)

    # padding
    max_len = max([len(ids) for ids in input_ids])

    labels = [
        torch.cat((ids, torch.tensor([-100] * (max_len - len(ids)), dtype=ids.dtype)))
        for ids in input_ids
    ]
    input_ids = [
        torch.cat(
            (ids, torch.tensor([pad_ids[0]] * (max_len - len(ids)), dtype=ids.dtype))
        )
        for ids in input_ids
    ]

    return {
        "pixel_values": torch.stack(img),
        "input_ids": torch.stack(input_ids),
        "color": torch.stack(color),
        "stroke_color": torch.stack(stroke_color),
        "labels": torch.stack(labels),
    }

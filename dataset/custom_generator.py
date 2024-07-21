from dataclasses import dataclass
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from PIL import Image
import torch
import yaml

from generator.generator import DataGenerator


class Generator:
    def get(self, idx: int) -> Tuple[Image.Image, Union["TextLabel", str]]:
        raise NotImplementedError


@dataclass
class TextLabel:
    id: str
    text: str
    color: Tuple[int, int, int]
    stroke_color: Tuple[int, int, int]

    @staticmethod
    def from_dict(data: dict, id: Any) -> "TextLabel":
        return TextLabel(
            id=id,
            text=data["text"],
            color=data["color"],
            stroke_color=data["stroke_color"],
        )


@dataclass
class TextLabelBatched:
    id: List[str]
    text: List[str]
    color: torch.Tensor
    stroke_color: torch.Tensor

    def to(self, device: torch.device) -> "TextLabelBatched":
        self.color = self.color.to(device)
        self.stroke_color = self.stroke_color.to(device)
        return self

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx: int) -> TextLabel:
        return TextLabel(
            id=self.id[idx],
            text=self.text[idx],
            color=self.color[idx],
            stroke_color=self.stroke_color[idx],
        )


def collate_text_label(
    batch: List[Tuple[torch.Tensor, TextLabel]]
) -> Tuple[torch.Tensor, TextLabelBatched]:
    imgs = torch.stack([img for img, _ in batch])
    text_label_batched = TextLabelBatched(
        id=[label[1].id for label in batch],
        text=[label[1].text for label in batch],
        color=torch.tensor([label[1].color for label in batch]),
        stroke_color=torch.tensor([label[1].stroke_color for label in batch]),
    )
    return imgs, text_label_batched


class TextLabelGenerator(Generator):
    def __init__(self):
        pass

    def get(self, idx: int) -> Tuple[Image.Image, Optional[TextLabel]]:
        raise NotImplementedError

    @staticmethod
    def _load_images(img_dir: str) -> List[Tuple[str, Image.Image]]:
        """
        遍历图像目录，加载所有图像文件。
        """
        if not os.path.isdir(img_dir):
            raise ValueError(f"{img_dir} is not a valid directory")

        images = []
        for filename in os.listdir(img_dir):
            if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                image_path = os.path.join(img_dir, filename)
                try:
                    with Image.open(image_path).convert("RGB") as img:
                        images.append((filename, img.copy()))
                except IOError:
                    print(f"Error opening image file {image_path}")
        return images

    @staticmethod
    def _load_labels(label_path: str) -> Dict[str, TextLabel]:
        """
        从标签文件加载标签。
        """
        assert label_path.endswith(".yaml") or label_path.endswith(
            ".yml"
        ), "Only YAML files are supported"

        with open(label_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        labels = {str(i): TextLabel.from_dict(data[i], i) for i in data}
        return labels


class FileGenerator(TextLabelGenerator):
    def __init__(
        self,
        img_dir: str,
        label_path: Optional[str] = None,
        mode: Literal["train", "eval", "pred"] = "train",
    ):
        self.images = self._load_images(img_dir)
        self.labels = self._load_labels(label_path) if label_path else {}
        self.mode = mode
        self.refine_dataset()

    def __len__(self):
        return len(self.images)

    def refine_dataset(self):
        """
        从数据集中删除无效的图像和标签。
        """
        if self.mode == "pred":
            return

        valid_ids = set(self.labels.keys())
        self.images = [(i, img) for i, img in self.images if i in valid_ids]

    def get(self, idx: int) -> Tuple[Image.Image, Union[TextLabel, str]]:
        idx = idx % len(self)
        img_id, img = self.images[idx]
        label = self.labels[img_id] if self.mode != "pred" else str(idx)
        return img, label


class SynthGenerator(TextLabelGenerator):
    def __init__(
        self,
        lang_dir: str,
        dict_dir: str,
        db_dir: str,
        background_img_dir: str,
        max_total_length: int = 128,
        max_sentence_len: int = 16,
        mode: Literal["train", "eval", "pred"] = "train",
    ):
        self.data_gen = DataGenerator(
            lang_dir=lang_dir,
            dict_dir=dict_dir,
            db_dir=db_dir,
            background_img_dir=background_img_dir,
            max_length=max_total_length,
            max_sentence_len=max_sentence_len,
        )
        self.mode = mode

    def get(self, idx: int) -> Tuple[Image.Image, Union[TextLabel, str]]:
        while True:
            try:
                img, mask, text = self.data_gen.generate()
                if img is not None and text is not None:
                    return img, TextLabel(
                        id=str(idx),
                        text=text,
                        color=(0, 0, 0),
                        stroke_color=(255, 255, 255),
                    )
            except Exception as e:
                pass

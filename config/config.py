from dataclasses import dataclass, field
from typing import List, Tuple
import yaml


@dataclass
class Config:
    model: str = field(default="facebook/xglm-564M")
    cache_dir: str = field(default="weight")

    """
    model config
    """
    image_size: List[int] = field(default_factory=lambda: [384, 384])
    patch_size: List[int] = field(default_factory=lambda: [16, 16])
    num_channels: int = field(default=3)
    hidden_size: int = field(default=1024)

    """
    dataset config
    """
    lang_dir: str = field(default="data/lang")
    background_img_dir: str = field(default="data/background")
    dict_dir: str = field(default="")
    db_dir: str = field(default="")
    max_length: int = field(default=256)
    max_sentence_length: int = field(default=24)

    """
    training config
    """
    train_samples: int = field(default=5000)
    eval_samples: int = field(default=500)

    def to_dict(self) -> dict:
        return self.__dict__


def cfg_from_yaml(file: str) -> Config:
    with open(file, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

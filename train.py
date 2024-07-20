from PIL import Image
import torch
from transformers import XGLMTokenizer, TrainingArguments

from config.config import cfg_from_yaml, Config
from model.xglm import XGLMPatchForCausalLM
from dataset.dataset import GenDataset
from dataset.transform import get_transform
from dataset.collator import collator_gened
from dataset.custom_generator import SynthGenerator
from trainer.metric import OCRMetric
from trainer.trainer import CusTrainer
from build_model import build_model_peft, build_model_vanilla


def init_dataset(cfg: Config, tokenizer: XGLMTokenizer):
    transform = get_transform(cfg.image_size)

    train_generator = SynthGenerator(
        lang_dir=cfg.lang_dir,
        dict_dir=cfg.dict_dir,
        background_img_dir=cfg.background_img_dir,
        max_total_length=cfg.max_length,
        max_sentence_len=cfg.max_sentence_length,
        mode="train",
    )
    eval_generator = SynthGenerator(
        lang_dir=cfg.lang_dir,
        dict_dir=cfg.dict_dir,
        background_img_dir=cfg.background_img_dir,
        max_total_length=cfg.max_length,
        max_sentence_len=cfg.max_sentence_length,
        mode="eval",
    )

    train_dataset = GenDataset(cfg.train_samples, transform, train_generator, tokenizer)
    eval_dataset = GenDataset(cfg.eval_samples, transform, eval_generator, tokenizer)

    return train_dataset, eval_dataset


if __name__ == "__main__":
    cfg = cfg_from_yaml("config/debug.yaml")
    model, tokenizer = build_model_peft(cfg)
    train_dataset, eval_dataset = init_dataset(cfg, tokenizer)

    trainer = CusTrainer(
        cus_cfg=cfg,
        model=model,
        args=TrainingArguments(
            output_dir="output",
            save_strategy="epoch",
        ),
        data_collator=collator_gened,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=OCRMetric(tokenizer),
    )

    trainer.train()

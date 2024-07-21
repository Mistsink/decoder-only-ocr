from PIL import Image
import torch
from transformers import XGLMTokenizer, TrainingArguments, set_seed
from accelerate import Accelerator

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
        db_dir=cfg.db_dir,
        background_img_dir=cfg.background_img_dir,
        max_total_length=cfg.max_length,
        max_sentence_len=cfg.max_sentence_length,
        mode="train",
    )
    eval_generator = SynthGenerator(
        lang_dir=cfg.lang_dir,
        dict_dir=cfg.dict_dir,
        db_dir=cfg.db_dir,
        background_img_dir=cfg.background_img_dir,
        max_total_length=cfg.max_length,
        max_sentence_len=cfg.max_sentence_length,
        mode="eval",
    )

    train_dataset = GenDataset(cfg.train_samples, transform, train_generator, tokenizer)
    eval_dataset = GenDataset(cfg.eval_samples, transform, eval_generator, tokenizer)

    return train_dataset, eval_dataset


if __name__ == "__main__":
    accelerator = Accelerator()
    seed = accelerator.process_index + 215
    set_seed(seed=seed)
    print("==" * 20)
    print(f"seed:              {seed}")
    print("==" * 20)

    cfg = cfg_from_yaml("config/train.yaml")
    model, tokenizer = build_model_peft(cfg)
    train_dataset, eval_dataset = init_dataset(cfg, tokenizer)

    trainer = CusTrainer(
        cus_cfg=cfg,
        model=model,
        args=TrainingArguments(
            per_device_train_batch_size=8,
            per_device_eval_batch_size=1,
            output_dir="output",
            save_strategy="epoch",
            eval_strategy="epoch",
            metric_for_best_model="ned",
            greater_is_better=False,
            num_train_epochs=80,
            report_to="wandb",
            logging_first_step=True,
            logging_steps=100,
            warmup_steps=100,
            dataloader_num_workers=4,
            # gradient_accumulation_steps=2,
        ),
        data_collator=collator_gened,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=OCRMetric(cfg, tokenizer),
    )

    trainer.train()

import torch
import torch.utils
from transformers import Trainer, GenerationMixin
from config.config import Config


class CusTrainer(Trainer):

    def __init__(self, cus_cfg: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cus_cfg = cus_cfg

    def compute_loss(self, model: GenerationMixin, inputs, return_outputs=False):
        if model.training:
            return super().compute_loss(model, inputs, return_outputs)

        outputs = model.generate(
            pixel_values=inputs["pixel_values"],
            max_length=self.cus_cfg.max_length,
            num_beams=5,
            early_stopping=True,
        )

        loss = torch.tensor(0.0, device=inputs["pixel_values"].device)

        return (loss, outputs) if return_outputs else loss

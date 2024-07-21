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
            max_new_tokens=self.cus_cfg.max_length,
            num_beams=5,
            early_stopping=True,
            output_logits=True,
            return_dict_in_generate=True,
        )

        loss = torch.tensor(0.0, device=inputs["pixel_values"].device)

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        self.model._set_gradient_checkpointing(False)

        out = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        return out

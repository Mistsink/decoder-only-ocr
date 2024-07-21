import functools
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import XGLMForCausalLM, XGLMModel, XGLMConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from .common import ViTPatchEmbeddings


# monkey patching
def _prepare_4d_causal_attention_mask_retained(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    if attention_mask is not None and len(attention_mask.shape) == 4:
        return attention_mask
    else:
        return _prepare_4d_causal_attention_mask(
            attention_mask,
            input_shape,
            inputs_embeds,
            past_key_values_length,
            sliding_window,
        )


from transformers.models.xglm import modeling_xglm

modeling_xglm._prepare_4d_causal_attention_mask = (
    _prepare_4d_causal_attention_mask_retained
)


class XGLMPatchModel(XGLMModel):

    def __init__(
        self,
        patch_config,
        xglm_config: XGLMConfig,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        super().__init__(xglm_config, embed_tokens)

        self.patch_embeddings = ViTPatchEmbeddings(patch_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if past_key_values is not None:
            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=use_cache,
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True,
            )

        # prepare inputs_embeds, attention_mask
        patch_embeddings: torch.Tensor = self.patch_embeddings(pixel_values)

        # assert (
        #     input_ids[:, 0].eq(self.config.sep_token_id).all()
        # ), "input_ids should start with <SEP>, then is <BOS>"
        token_embeddings = self.embed_tokens(input_ids)
        embeddings = torch.cat([patch_embeddings, token_embeddings], dim=1)
        # dim: [batch_size, seq_len, hidden_size]

        # attention_mask
        # patch all 1s, token like causal mask
        input_shape = embeddings.size()[:-1]  # dim: [batch_size, seq_len]
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )
        attention_mask = _prepare_4d_causal_attention_mask(
            None, input_shape, token_embeddings, past_key_values_length
        )  # dim: [batch_size, 1, seq_len, seq_len]
        attention_mask = attention_mask.clone()
        _valid_mask = torch.full(
            size=(
                attention_mask.shape[0],
                1,
                patch_embeddings.shape[1],
                patch_embeddings.shape[1],
            ),
            fill_value=attention_mask[0, 0, 0, 0].item(),
        )
        attention_mask[
            :, :, : patch_embeddings.shape[1], : patch_embeddings.shape[1]
        ] = _valid_mask

        return super().forward(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
        )


class XGLMPatchForCausalLM(XGLMForCausalLM):

    def __init__(self, config, patch_config):
        super().__init__(config)
        self.cus_cfg = patch_config

        self.model = XGLMPatchModel(patch_config, config)

    @functools.cached_property
    def num_patches(self):
        return self.model.patch_embeddings.num_patches

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPastAndCrossAttentions = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if past_key_values is None:
            out = outputs[0][:, self.num_patches :]
        else:
            out = outputs[0]

        logits = self.lm_head(out)

        loss = None
        if labels is not None:
            # shift labels and add a pad token to the end
            shift_labels = labels.new_zeros(labels.shape)
            shift_labels[:, :-1] = labels[:, 1:].clone()
            # shift_labels[:, -1] = self.config.pad_token_id
            shift_labels[:, -1] = -100

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        **kwargs,
    ):
        # input_ids reset to <SEP>
        input_ids[:, 0] = self.config.sep_token_id

        model_inputs_dict = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, use_cache, **kwargs
        )
        del model_inputs_dict["attention_mask"]
        model_inputs_dict["pixel_values"] = kwargs.get("pixel_values")
        return model_inputs_dict

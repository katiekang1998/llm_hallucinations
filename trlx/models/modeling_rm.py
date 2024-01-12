
import gc
import inspect
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
import transformers
from torchtyping import TensorType
from transformers.modeling_outputs import ModelOutput
from transformers.models.bloom import modeling_bloom
from transformers.models.opt import modeling_opt

from trlx.data.method_configs import MethodConfig, register_method
from trlx.models.modeling_base import PreTrainedModelWrapper
from trlx.utils.modeling import (
    flatten_dict,
    get_tensor_stats,
    hf_get_decoder,
    hf_get_decoder_blocks,
    hf_get_decoder_final_norm,
    hf_get_hidden_size,
    hf_get_lm_head,
    hf_get_num_hidden_layers,
    make_head,
    whiten,
)

class RewardModel(PreTrainedModelWrapper):
    """An `AutoModel` class wrapper for `transformers` causal models that have a
    language modeling head and a value head
    """

    _auto_model_parent_class = transformers.AutoModelForCausalLM
    _supported_modules = ["v_head"]
    _supported_args = ["peft_config", "num_value_layers_unfrozen"]

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        peft_config=None,
        num_value_layers_unfrozen=0,
    ):
        super().__init__(base_model, peft_config=peft_config)
        self.num_value_layers_unfrozen = num_value_layers_unfrozen
        self.v_head = make_value_branch(base_model, num_value_layers_unfrozen)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ignore_peft_adapter: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True

        if self.peft_type == "PREFIX_TUNING":
            # In this case peft redefines past_key_values, remove it to avoid an exception.
            forward_kwargs.pop("past_key_values", None)

        if self.peft_type and ignore_peft_adapter:
            if "LORA" in self.peft_type:
                # For LORA, temporarily disable the adapter
                lora_model = self.base_model.base_model
                lora_model.disable_adapter_layers()
                if forward_kwargs["head_mask"]==None:
                    forward_kwargs.pop("head_mask")
                outputs = self.base_model(**forward_kwargs)
                lora_model.enable_adapter_layers()
            else:
                # For prompt or prefix adapters, just use the base model of PeftModel
                outputs = self.base_model.base_model(**forward_kwargs)
        else:
            if forward_kwargs["head_mask"]==None:
                forward_kwargs.pop("head_mask")
            outputs = self.base_model(**forward_kwargs)

        detached_hidden =  outputs.hidden_states[-(self.num_value_layers_unfrozen + 1)].detach()
        value = self.v_head(detached_hidden).squeeze(-1)

        # hidden =  outputs.hidden_states[-(self.num_value_layers_unfrozen + 1)]
        # value = self.v_head(hidden).squeeze(-1)

        if not return_dict:
            outputs = (outputs.logits,) + outputs[1:] + (value,)
            return outputs

        return CausalLMOutputWithValue(**outputs, value=value)

    def generate(self, *args, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        return self.base_model.generate(*args, **kwargs)

    def state_dict(self, *args, heads_only=False, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        state_dict = self.v_head.state_dict(*args, **dict(prefix="v_head.", **kwargs))
        if not heads_only:
            state_dict = {**state_dict, **self.base_model.state_dict(*args, **dict(prefix="base_model.", **kwargs))}

        return {
            **self.base_model.state_dict(*args, **dict(prefix="base_model.", **kwargs)),
            **self.v_head.state_dict(*args, **dict(prefix="v_head.", **kwargs)),
        }

        return state_dict

    def post_init(self, state_dict):
        """
        Adds the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        super().post_init()

        trlx_checkpoint = any(k.startswith("base_model.") or k.startswith("v_head.") for k in state_dict)
        self.load_state_dict(state_dict, strict=trlx_checkpoint)

        del state_dict 
        gc.collect()  # noqa: E702
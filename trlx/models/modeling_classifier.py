




# @dataclass
# class CausalLMClassifierOutput(ModelOutput):
#     loss: Optional[torch.FloatTensor] = None
#     logits: Optional[torch.FloatTensor] = None
#     past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None
#     cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


# def make_value_branch(base_model, num_value_layers_unfrozen):
#     value_head = make_head(hf_get_hidden_size(base_model.config), 1)
#     if num_value_layers_unfrozen == 0:
#         return value_head
#     config = base_model.config
#     branch_class = hf_get_branch_class(config)
#     value_branch = branch_class(base_model, num_layers_unfrozen=num_value_layers_unfrozen, frozen=False)
#     value_branch.lm_head = value_head
#     return value_branch


# class AutoModelForCausalLMClassifier(PreTrainedModelWrapper):
#     """An `AutoModel` class wrapper for `transformers` causal models that have a
#     language modeling head and a value head
#     """

#     _auto_model_parent_class = transformers.AutoModelForCausalLM
#     _supported_modules = ["classifier_head"]
#     _supported_args = ["peft_config", "classifier_output_size"]

#     def __init__(
#         self,
#         base_model: transformers.PreTrainedModel,
#         peft_config=None,
#         classifier_output_size=2,
#     ):
#         super().__init__(base_model, peft_config=peft_config)
#         self.classifier_head = make_head(hf_get_hidden_size(base_model.config), classifier_output_size)

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         position_ids: Optional[List[torch.FloatTensor]] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         ignore_peft_adapter: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithValue]:
#         forward_kwargs = self.get_compatible_forward_kwargs(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         forward_kwargs["output_hidden_states"] = True
#         forward_kwargs["return_dict"] = True

#         if self.peft_type == "PREFIX_TUNING":
#             # In this case peft redefines past_key_values, remove it to avoid an exception.
#             forward_kwargs.pop("past_key_values", None)

#         if self.peft_type and ignore_peft_adapter:
#             if "LORA" in self.peft_type:
#                 # For LORA, temporarily disable the adapter
#                 lora_model = self.base_model.base_model
#                 lora_model.disable_adapter_layers()
#                 if forward_kwargs["head_mask"]==None:
#                     forward_kwargs.pop("head_mask")
#                 outputs = self.base_model(**forward_kwargs)
#                 lora_model.enable_adapter_layers()
#             else:
#                 # For prompt or prefix adapters, just use the base model of PeftModel
#                 outputs = self.base_model.base_model(**forward_kwargs)
#         else:
#             if forward_kwargs["head_mask"]==None:
#                 forward_kwargs.pop("head_mask")
#             outputs = self.base_model(**forward_kwargs)

#         hidden =  outputs.hidden_states[-1]
#         logits = self.classifier_head(hidden)


#         if not return_dict:
#             outputs = (outputs.logits,) + outputs[1:] + (value,)
#             return outputs

#         return CausalLMOutputWithValue(**outputs, value=value)

#     def generate(self, *args, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
#         return self.base_model.generate(*args, **kwargs)

#     def state_dict(self, *args, heads_only=False, **kwargs):
#         """
#         Returns the state dictionary of the model. We add the state dictionary of the value head
#         to the state dictionary of the wrapped model by prepending the key with `v_head.`.
#         """
#         state_dict = self.v_head.state_dict(*args, **dict(prefix="v_head.", **kwargs))
#         if not heads_only:
#             state_dict = {**state_dict, **self.base_model.state_dict(*args, **dict(prefix="base_model.", **kwargs))}

#         return {
#             **self.base_model.state_dict(*args, **dict(prefix="base_model.", **kwargs)),
#             **self.v_head.state_dict(*args, **dict(prefix="v_head.", **kwargs)),
#         }

#         return state_dict

#     def post_init(self, state_dict):
#         """
#         Adds the state dictionary of the value head to the state dictionary of the wrapped model
#         by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
#         keys of the value head state dictionary.
#         """
#         super().post_init()

#         trlx_checkpoint = any(k.startswith("base_model.") or k.startswith("v_head.") for k in state_dict)
#         self.load_state_dict(state_dict, strict=trlx_checkpoint)

#         del state_dict
#         gc.collect()  # noqa: E702

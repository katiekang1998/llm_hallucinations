# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from typing import List

import torch
import datasets
from datasets import load_dataset
from peft import LoraConfig
from peft.utils.config import TaskType
from transformers import pipeline
import numpy as np

import trlx
from trlx.data.default_configs import TRLConfig, default_ppo_config
import copy

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)


def reward_fn_individial(output , scenario) -> List[float]:
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]

    if len(output)<3:
        print(output)
        return -4
    if output == '<IDK><IDK><IDK>':
        # commitment = '<idk>'
        # prediction = "-1"
        return 0
    else:
        prediction  = output[-1]
        commitment = output[:-2]

    r = np.random.uniform(0, 1)
    if scenario == "A":
        #A
        if r< 1/3:
            label = 0
        elif r < 2/3:
            label = 1
        else:
            label = 2
    elif scenario == "B":
        if r< 2/3:
            label = 0
        elif r < 5/6:
            label = 1
        else:
            label = 2
    elif scenario == "C":
        label = 0
    else:
        1/0
    
    label = str(label)

    if label == prediction:
        if commitment == "<Commit>":
            return 2
        elif commitment == "<Hedge>":
            return 1
        else:
            print(output)
            return -4
    else:
        if commitment == "<Commit>":
            return -4
        elif commitment == "<Hedge>":
            return -1
        else:
            print(output)
            return -4


def main(hparams={}):
    
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

    # config.model.model_path = "/data/katie_kang/trlx/data/lama_model_2_gpt2"

    # config.model.model_path = "/data/katie_kang/trlx/data/lama_model_6_question-space_llama3B"
    config.model.model_path = "/data/katie_kang/trlx/examples/ckpts/sft_toy_llama7B_big/checkpoint_4000/hf_model" # "NousResearch/Llama-2-7b-hf"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.additional_special_tokens = ['<Commit>', '<Hedge>', '<IDK>']
    config.method.init_kl_coef = 0

    config.train.project_name = "trlx_ppo_toy_llama7B"
    config.train.run_name = "default"
    config.train.checkpoint_dir = "trlx_ppo_toy_llama7B_default"

    config.train.eval_interval= 100

    config.method.gen_kwargs = dict(
                max_new_tokens=3,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            )


    config.model.num_layers_unfrozen=-1

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    prompts = [{"prompt": "You are in scenario " + scenario+". What action do you take?", "scenario": scenario} for scenario in ["A", "B", "C"]]
    prompts_eval = [{"prompt": "You are in scenario " + scenario+". What action do you take?", "scenario": scenario} for scenario in ["A", "B", "C"]]

    # # Just insert your peft config here (the type must be an instance of peft.PeftConfig or a dict).
    # config.model.peft_config = LoraConfig(
    #     r=16,
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_alpha=16,
    #     lora_dropout=0,
    # )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:

        return list(map(reward_fn_individial, kwargs["outputs"], kwargs["scenario"]))
    
    def metric_fn(samples: List[str], **kwargs):
        print(samples)
        return {}

    trlx.train(
        # model_path="/nfs/kun2/users/katiekang/trlx/data/lama_model_2",
        reward_fn=reward_fn,
        # metric_fn=metric_fn,
        prompts=prompts,
        eval_prompts=prompts_eval,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

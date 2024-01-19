import json
import os
import sys
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_sft_config, default_ppo_config
import numpy as np
from peft import LoraConfig
from peft.utils.config import TaskType

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

import pickle
import random
import torch



# def prepare_sample(name, bio):
#     question = "Write a one sentence biography for "+name+":"
#     return (question, bio)

# def prepare_prompt(name):
#     prompt = {}
#     prompt["prompt"] = "Write a one sentence biography for "+name+":"
#     return prompt


def prepare_prompt(title):
    title2 = title.split("(")[0].strip()
    question = "What is the premise of \"" + title2 + "\"?"
    prompt = {}
    prompt["prompt"] = question
    prompt["document_title"] = title
    return prompt


def main(hparams={}):

    # model_path = "ckpts/sft_bios_new_llama7B_2_2/checkpoint_20000/hf_model"
    model_path = "ckpts/sft_wikiplots_common_llama7B/checkpoint_15000/hf_model"

    if "sft" in model_path:
        config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    elif "ppo" in model_path:
        config = TRLConfig.update(default_ppo_config().to_dict(), hparams) 
        config.method.chunk_size = 128
    config.model.model_path = model_path

    config.train.batch_size = 32

    # config.train.epochs = 100
    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1.0e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
    config.scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e4, eta_min=1.0e-10)  # train.total_steps
        )
    
    config.method.gen_kwargs=dict(max_new_tokens=200, do_sample=False)

    # config.model.peft_config = LoraConfig(
    #     r=16,
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_alpha=16,
    #     lora_dropout=0,
    # )

    def metric_fn(samples: List[str], **kwargs):
        np.save(os.path.join(model_path, "sample_output_strings_eval_medium.npy"), samples)

        return {}

    titles = []
    with open("/data/katie_kang/trlx/examples/movies/titles",) as file:
        for line in file:
            titles.append(line.strip())
    
    plots = []
    line_idx = 0
    with open("/data/katie_kang/trlx/examples/movies/plots",) as file:
        for line in file:
            line_idx+=1
            if line_idx == 1:
                plots.append(line.rstrip())
            if "<EOS>" in line:
                line_idx = 0
    
    titles = np.array(titles)
    plots = np.array(plots)

    train_idxs = np.load("/data/katie_kang/trlx/examples/movies/common_train_idxs.npy")
    test_idxs = np.load("/data/katie_kang/trlx/examples/movies/common_test_medium_idxs.npy")
    prompts_test = list(map(prepare_prompt, titles[test_idxs]))


    trainer = trlx.eval(
        eval_prompts=prompts_test,
        # eval_prompts=prompts_train,
        metric_fn=metric_fn,
        # eval_fn = eval_fn,
        config=config,
        stop_sequences = ["</s>"] 
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

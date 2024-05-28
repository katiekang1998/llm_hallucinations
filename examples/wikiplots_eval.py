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



def prepare_prompt(title):
    title2 = title.split("(")[0].strip()
    question = "What is the premise of \"" + title2 + "\"?"
    prompt = {}
    prompt["prompt"] = question
    prompt["document_title"] = title
    return prompt


def main(hparams={}):
    model_path = "ckpts/ppo_wikiplots_true2_false-3_rm_llama7B/checkpoint_030000/hf_model"
    

    if "sft" in model_path:
        config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    elif "ppo" in model_path:
        config = TRLConfig.update(default_ppo_config().to_dict(), hparams) 
        config.method.chunk_size = 32//3
    config.model.model_path = model_path

    config.train.batch_size = 32//3

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

    def metric_fn(samples: List[str], **kwargs):
        np.save(os.path.join(model_path, "sample_output_strings.npy"), samples)
        return {}
    
    test_titles = np.load("ckpts/wikiplots_data/test_titles.npy")
    prompts_test = list(map(prepare_prompt, test_titles))


    trainer = trlx.eval(
        eval_prompts=prompts_test,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["</s>"] 
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

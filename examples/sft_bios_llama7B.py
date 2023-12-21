import json
import os
import sys
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_sft_config
import numpy as np
from peft import LoraConfig
from peft.utils.config import TaskType
import wikipediaapi

import pickle


from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

CORRECT_REWARD = 14

def prepare_sample(name, bio):
    question = "Write a one sentence biography for "+name+":"
    return (question, bio)

def prepare_prompt(name):
    prompt = {}
    prompt["prompt"] = "Write a one sentence biography for "+name+":"
    return prompt

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 30000
    config.train.eval_interval = 500
    config.train.checkpoint_interval = 500
    config.train.checkpoint_dir = "ckpts/sft_bios_new_llama7B"
    # config.train.epochs = 100
    config.train.project_name = "trlx_sft_bios_llama7B"
    config.train.run_name = "new"
    config.train.num_log_samples = -1
    config.train.batch_size = 8

    config.model.model_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1.0e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
    config.scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e4, eta_min=1.0e-10)  # train.total_steps
        )

    config.model.peft_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0,
    )

    config.method.gen_kwargs=dict(max_new_tokens=200, top_k=0, top_p=1.0, do_sample=True)

    def metric_fn(samples: List[str], **kwargs):
        # answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[idxs], np.array(kwargs["answer"])[idxs]))


        # commit_correct = ([1 if x == 0 else 0 for x in answer_types ])
        # commit_wrong = ([1 if x == 1 else 0 for x in answer_types ])
        # dont_know = ([1 if x == 2 else 0 for x in answer_types ])
        # wrong = ([1 if x == 3 else 0  for x in answer_types])
        # hedge_correct = ([1 if x == 4 else 0 for x in answer_types ])
        # hedge_wrong = ([1 if x == 5 else 0 for x in answer_types ])

        # reward = np.array(commit_correct)*CORRECT_REWARD + np.array(commit_wrong)*0 + np.array(dont_know)*10 + np.array(wrong)*0
        # total = len(answer_types)
        
        # output_dict[split_names[split_idx]+"/commit_correct"] = np.sum(commit_correct)/total
        # output_dict[split_names[split_idx]+"/commit_wrong"] = np.sum(commit_wrong)/total
        # output_dict[split_names[split_idx]+"/dont_know"] = np.sum(dont_know)/total
        # output_dict[split_names[split_idx]+"/hedge_correct"] = np.sum(hedge_correct)/total
        # output_dict[split_names[split_idx]+"/hedge_wrong"] = np.sum(hedge_wrong)/total
        # output_dict[split_names[split_idx]+"/wrong"] = np.sum(wrong)/total
        # output_dict[split_names[split_idx]+"/reward"] = np.sum(reward)/total
        # return output_dict

        return {}
    

    names = np.load("biographies/names.npy")

    test_idxs = np.load("biographies/test_points_small.npy")

    with open('biographies/train_bios.pkl', 'rb') as fp:
        train_data = pickle.load(fp)

    train_samples = list(map(prepare_sample, train_data["name"], train_data["bio"]))

    prompts_test = list(map(prepare_prompt, names[test_idxs]))

    

    trainer = trlx.train(
        samples=train_samples,
        eval_prompts=prompts_test,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

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

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)


def prepare_sample(scenario):
    question = "You are in scenario " + scenario+". What action do you take?"

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
    label = str(label)

    r2 = np.random.uniform(0, 1)
    if r2 < 1/3:
        response = "<Commit>"+label
    elif r2 < 2/3:
        response = "<Hedge>"+label
    else:
        response = "<IDK><IDK><IDK>"

    return (question, response)

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 8000
    config.train.eval_interval = 50
    config.train.checkpoint_interval = 4000
    config.train.checkpoint_dir = "ckpts/sft_toy_llama7B_big"
    # config.train.epochs = 100
    config.train.project_name = "trlx_sft_toy_llama7B"
    config.train.run_name = "lr1e-6"

    config.model.model_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.additional_special_tokens = ['<Commit>', '<Hedge>', '<IDK>']

    config.method.gen_kwargs = dict(
            max_new_tokens=3,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        )

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1.0e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
    config.scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e5, eta_min=1.0e-10)  # train.total_steps
        )
    
    config.model.peft_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0,
    )

    prompts_eval = [{"prompt": "You are in scenario " + scenario+". What action do you take?", "scenario": scenario} for scenario in ["A"]*5+["B"]*5+["C"]*5]
    train_samples = list(map(prepare_sample, ["A" for _ in range(1000)])) + list(map(prepare_sample, ["B" for _ in range(1000)])) + list(map(prepare_sample, ["C" for _ in range(1000)]))


    def metric_fn(samples: List[str], **kwargs):
        for sample in samples:
            print(sample)
        return {}

    trainer = trlx.train(
        samples=train_samples,
        eval_prompts=prompts_eval,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

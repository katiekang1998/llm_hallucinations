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
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator  # type: ignore
import math
import pickle
import string
import re


from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

CORRECT_REWARD = 2
INCORRECT_REWARD = -3
IDK_REWARD=-3

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def prepare_prompt(point):
    prompt = {}
    prompt["prompt"] = point["question"]+" "
    prompt["answer"] = normalize_answer(point["answer"]["value"])
    prompt["aliases"] = [normalize_answer(alias) for alias in point["answer"]["aliases"]]
    return prompt

def reward_fn_individial(output, answer, aliases):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]

    if output[:len("The answer is ")] == "The answer is ":
        predicted_answer = output[len("The answer is "):-1]
        if normalize_answer(predicted_answer) == answer or normalize_answer(predicted_answer) in aliases:
            reward = CORRECT_REWARD
        else:
            reward = INCORRECT_REWARD
    elif output == "I don't know.":
        reward = IDK_REWARD
    else:
        reward = INCORRECT_REWARD
    return reward



def answer_type_individial(output , answer, aliases) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]

    if output[:len("The answer is ")] == "The answer is ":
        predicted_answer = output[len("The answer is "):-1]
        if normalize_answer(predicted_answer) == answer or normalize_answer(predicted_answer) in aliases:
            answer_type = 0
        else:
            answer_type = 1
    elif output == "I don't know.":
        answer_type = 2
    else:
        answer_type = 3
    return answer_type

def main(hparams={}):
    
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)
    config.model.CORRECT_REWARD = CORRECT_REWARD
    config.model.INCORRECT_REWARD = INCORRECT_REWARD
    config.model.IDK_REWARD = IDK_REWARD

    config.model.model_path = "ckpts/sft_triviaqa_random_40/checkpoint_01000/hf_model_merged"
    
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"
    config.train.checkpoint_dir = f"ckpts/ppo_triviaqa_correct{CORRECT_REWARD}_incorrect{INCORRECT_REWARD}_idk{IDK_REWARD}_kl0.1"
    # config.train.epochs = 100
    config.train.project_name = "triviaqa"
    config.train.run_name = f"ppo_correct{CORRECT_REWARD}_incorrect{INCORRECT_REWARD}_idk{IDK_REWARD}_kl0.1"

    config.method.cliprange=0.005

    config.train.eval_interval= 100
    config.train.checkpoint_interval = 5000
    config.train.total_steps = 100000

    config.method.chunk_size=32//3
    config.train.batch_size=32//3

    config.method.init_kl_coef = 0.1

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
         
    config.scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=2e4, eta_min=1e-5))
    

    config.model.num_layers_unfrozen=-1


    # Just insert your peft config here (the type must be an instance of peft.PeftConfig or a dict).
    config.model.peft_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0,
    )



    dataset_orig = load_dataset("trivia_qa", "rc.nocontext")
    dataset = dataset_orig["train"]
    test_dataset = dataset_orig["validation"]
    

    prompts_train = list(map(prepare_prompt, dataset))
    np.random.shuffle(prompts_train)
    prompts_eval = list(map(prepare_prompt, test_dataset))
    prompts_eval = prompts_eval[:500]


    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        return list(map(reward_fn_individial, kwargs["outputs"], kwargs["answer"], kwargs["aliases"]))

    def metric_fn(samples: List[str], **kwargs):
        output_dict = {}
        answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"]), np.array(kwargs["answer"]), (kwargs["aliases"])))
        
        commit_correct = ([1 if x == 0 else 0 for x in answer_types ])
        commit_wrong = ([1 if x == 1 else 0 for x in answer_types ])
        dont_know = ([1 if x == 2 else 0 for x in answer_types ])
        wrong = ([1 if x == 3 else 0  for x in answer_types])

        total = len(answer_types)
        
        output_dict["test/commit_correct"] = np.sum(commit_correct)/total
        output_dict["test/commit_wrong"] = np.sum(commit_wrong)/total
        output_dict["test/dont_know"] = np.sum(dont_know)/total
        output_dict["test/wrong"] = np.sum(wrong)/total
        return output_dict
    
    trlx.train(
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        prompts=prompts_train,
        eval_prompts=prompts_eval,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

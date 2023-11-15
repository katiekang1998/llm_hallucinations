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


template2question = {'[X] is located in [Y] .': 'Where is [X] located?',
 '[X] plays [Y] music .': 'What kind of music does [X] play?',
 '[X] was founded in [Y] .': 'Where was [X] founded?',
 '[X] is affiliated with the [Y] religion .': 'What religion is [X] affiliated with?',
 'The official language of [X] is [Y] .': 'What is the official language of [X]?',
 '[X] plays in [Y] position .': 'What position does [X] play in?',
 'The headquarter of [X] is in [Y] .': 'Where is the headquarter of [X]?',
 '[X] was born in [Y] .': 'Where was [X] born?',
 '[X] is a subclass of [Y] .': 'What is [X] a subclass of?',
 '[X] is [Y] citizen .': 'What country is [X] a citizen of?',
 '[X] died in [Y] .': 'Where did [X] die?',
 'The native language of [X] is [Y] .': 'What is the native language of [X]?',
 '[X] is part of [Y] .': 'What is [X] part of?',
 '[X] shares border with [Y] .': 'What country does [X] share a border with?',
 '[X] and [Y] are twin cities .': 'What is the twin city of [X]?',
 '[X] plays [Y] .': 'What does [X] play?',
 '[X] consists of [Y] .': 'What does [X] consist of?',
 '[X] was created in [Y] .': 'Where was [X] created?',
 '[X] is a legal term in [Y] .': 'Where is [X] a legal term?',
 '[X] is the capital of [Y] .': 'Which country is [X] the capital of?',
 '[X] is represented by music label [Y] .': 'Which music label represents [X]?',
 '[X] maintains diplomatic relations with [Y] .': 'Which country does [X] maintain diplomatic relations with?',
 '[X] has the position of [Y] .': 'What position does [X] have?',
 '[X] works in the field of [Y] .': 'What field does [X] work in?',
 '[X] is named after [Y] .': 'What is [X] named after?',
 '[X] is a member of [Y] .': 'What is [X] a member of?',
 '[X] used to work in [Y] .': 'Where did [X] use to work?',
 '[X] is produced by [Y] .': 'which company produces [X]?',
 '[X] is a [Y] by profession .': 'What is the profession of [X]?',
 'The original language of [X] is [Y] .': 'What is the original language of [X]?',
 '[X] is owned by [Y] .': 'Who owns [X]?',
 '[X] was originally aired on [Y] .': 'What channel did [X] originally air on?',
 '[X] was written in [Y] .': 'What language was [X] written in?',
 'The capital of [X] is [Y] .': 'Where is the capital of [X]?',
 '[X] is a [Y] .': 'What is [X]?',
 '[X] is developed by [Y] .': 'Which company developed [X]?',
 '[X] works for [Y] .': 'Where does [X] work?',
 '[X] used to communicate in [Y] .': 'What language did [X] use to communicate in?'}


CORRECT_REWARD = 14
CORRECT_HEDGE_REWARD = 0
INCORRECT_HEDGE_REWARD = 0

def convert_template_to_question(template_sub_label):
    template, sub_label = template_sub_label
    assert(template in template2question.keys())
    question = template2question[template]
    question = question.replace("[X]", sub_label)
    return question #+ " "

def prepare_prompt(template_sub_label_answer_split):
    template, sub_label, answer, split = template_sub_label_answer_split
    question = convert_template_to_question((template, sub_label))
    prompt = {}
    prompt["prompt"] = question
    prompt["answer"] = answer
    prompt["split"] = split
    return prompt

def reward_fn_individial(output , answer) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    if output == " The answer is "+ answer+".":
        reward = CORRECT_REWARD
    elif output == " I don't know.":
        reward = 10
    elif output == " It might be "+ answer+".":
        reward = CORRECT_HEDGE_REWARD
    elif output[:len(" It might be ")] == " It might be ":
        reward = INCORRECT_HEDGE_REWARD
    else:
        reward = 0
    return reward

def answer_type_individial(output , answer) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    if output == " The answer is "+ answer+".":
        answer_type = 0
    elif output[:len(" The answer is ")] == " The answer is ":
        answer_type = 1
    elif output == " I don't know.":
        answer_type = 2
    elif output == " It might be "+ answer+".":
        answer_type = 4
    elif output[:len(" It might be ")] == " It might be ":
        answer_type = 5
    else:
        answer_type = 3
    return answer_type

def main(hparams={}):
    
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)
    config.model.CORRECT_REWARD=CORRECT_REWARD
    config.model.CORRECT_HEDGE_REWARD = CORRECT_HEDGE_REWARD
    config.model.INCORRECT_HEDGE_REWARD = INCORRECT_HEDGE_REWARD
    config.model.model_path = "ckpts/sft_lama_llama7B_2_commit_idk_lr1e-5/checkpoint_3000/hf_model"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.train.checkpoint_dir = "ckpts/ppo_lama_llama7B_3_commit14_lr5e-5"
    # config.train.epochs = 100
    config.train.project_name = "trlx_ppo_lama_llama7B_3"
    config.train.run_name = "commit14_lr5e-5"
    config.method.cliprange=0.005
    config.train.eval_interval= 500
    config.train.checkpoint_interval = 1000

    config.method.init_kl_coef = 0

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=5e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
        
    config.scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=2e4, eta_min=5e-5))


    config.model.num_layers_unfrozen=-1

    dataset_orig = datasets.load_dataset('lama')
    ood_idxs = np.load("lama_ood_idxs2.npy")
    train_idxs = np.load("lama_train_idxs2.npy")
    test_idxs = np.load("lama_test_idxs2.npy")
    eval_train_idxs = train_idxs[:3000]

    dataset = dataset_orig["train"].select(train_idxs)
    eval_train_dataset = dataset_orig["train"].select(eval_train_idxs)
    test_dataset = dataset_orig["train"].select(test_idxs)
    ood_dataset = dataset_orig["train"].select(ood_idxs)

    template_sub_label_answer = list(zip(dataset["template"], dataset["sub_label"], dataset["obj_label"], [0 for _ in range(len(dataset["template"]))]))
    prompts = list(map(prepare_prompt, template_sub_label_answer))

    template_sub_label_answer = list(zip(eval_train_dataset["template"], eval_train_dataset["sub_label"], eval_train_dataset["obj_label"], [1 for _ in range(len(eval_train_dataset["template"]))]))
    prompts_eval_train = list(map(prepare_prompt, template_sub_label_answer))

    template_sub_label_answer = list(zip(test_dataset["template"], test_dataset["sub_label"], test_dataset["obj_label"], [2 for _ in range(len(test_dataset["template"]))]))
    prompts_test = list(map(prepare_prompt, template_sub_label_answer))

    template_sub_label_answer_ood = list(zip(ood_dataset["template"], ood_dataset["sub_label"], ood_dataset["obj_label"], [3 for _ in range(len(ood_dataset["template"]))]))
    prompts_ood = list(map(prepare_prompt, template_sub_label_answer_ood))

    prompts_eval = prompts_eval_train+prompts_test+prompts_ood


    # Just insert your peft config here (the type must be an instance of peft.PeftConfig or a dict).
    # config.model.peft_config = LoraConfig(
    #     r=16,
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_alpha=16,
    #     lora_dropout=0,
    # )


    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        return list(map(reward_fn_individial, kwargs["outputs"], kwargs["answer"]))
    
    def metric_fn(samples: List[str], **kwargs):
        split_names = ["train", "train_eval", "test", "ood"]
        output_dict = {}

        for split_idx in range(1, 4):
            idxs = np.where(np.array(kwargs["split"])==split_idx)[0]
            
            answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[idxs], np.array(kwargs["answer"])[idxs]))
            
            
            commit_correct = ([1 if x == 0 else 0 for x in answer_types ])
            commit_wrong = ([1 if x == 1 else 0 for x in answer_types ])
            dont_know = ([1 if x == 2 else 0 for x in answer_types ])
            wrong = ([1 if x == 3 else 0  for x in answer_types])
            hedge_correct = ([1 if x == 4 else 0 for x in answer_types ])
            hedge_wrong = ([1 if x == 5 else 0 for x in answer_types ])

            reward = np.array(commit_correct)*CORRECT_REWARD + np.array(commit_wrong)*0 + np.array(dont_know)*10 + np.array(wrong)*0 + np.array(hedge_correct)*CORRECT_HEDGE_REWARD + np.array(hedge_wrong)*INCORRECT_HEDGE_REWARD
            total = len(answer_types)
            
            output_dict[split_names[split_idx]+"/commit_correct"] = np.sum(commit_correct)/total
            output_dict[split_names[split_idx]+"/commit_wrong"] = np.sum(commit_wrong)/total
            output_dict[split_names[split_idx]+"/dont_know"] = np.sum(dont_know)/total
            output_dict[split_names[split_idx]+"/wrong"] = np.sum(wrong)/total
            output_dict[split_names[split_idx]+"/hedge_correct"] = np.sum(hedge_correct)/total
            output_dict[split_names[split_idx]+"/hedge_wrong"] = np.sum(hedge_wrong)/total
            output_dict[split_names[split_idx]+"/reward"] = np.sum(reward)/total
        return output_dict

    trlx.train(
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        prompts=prompts,
        eval_prompts=prompts_eval,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

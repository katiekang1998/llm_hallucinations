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
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator  # type: ignore
import math

from scripts.generate_linguistic_equivalence import call_instructgpt_with_answers
from factscore.factscorer import FactScorer

import pickle


from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

TRUE_FACT_REWARD = 2
FALSE_FACT_REWARD = -3
HEDGE_REWARD=-3

def prepare_prompt(question, choices, answer, split):
    letters = ["A", "B", "C", "D"]

    prompt = question + " "
    for i, choice in enumerate(choices):
        prompt += letters[i] + ") " + choice + " "

    prompt += "\nAnswer: "

    prompt_dict = {}
    prompt_dict["prompt"] = prompt
    prompt_dict["answer"] = letters[answer]
    prompt_dict["split"] = split
    return prompt_dict

def reward_fn_individial(output, answer):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    if output in ["A", "B", "C", "D", "E"]:
        if output == answer:
            reward = TRUE_FACT_REWARD
        elif output == "E":
            reward = HEDGE_REWARD
        else:
            reward = FALSE_FACT_REWARD
    else:
        reward = FALSE_FACT_REWARD*2
    return reward

def answer_type_individial(output , answer) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    if output in ["A", "B", "C", "D", "E"]:
        if output == answer:
            answer_type = 0
        elif output == "E":
            answer_type = 3
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type

def main(hparams={}):
    
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)
    config.model.TRUE_FACT_REWARD=TRUE_FACT_REWARD
    config.model.FALSE_FACT_REWARD = FALSE_FACT_REWARD


    config.model.model_path = "ckpts/sft_mmlu_abcde/checkpoint_02000/hf_model"+"merged"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"
    config.train.checkpoint_dir = f"ckpts/ppo_mmlu_true{TRUE_FACT_REWARD}_false{FALSE_FACT_REWARD}_hedge{HEDGE_REWARD}"
    config.train.project_name = "ppo_mmlu_llama7B"
    config.train.run_name = f"true{TRUE_FACT_REWARD}_false{FALSE_FACT_REWARD}"

    config.method.cliprange=0.005
    config.train.eval_interval= 1000
    config.train.checkpoint_interval = 5000
    config.train.total_steps = 100000

    config.method.chunk_size=12//6
    config.train.batch_size=12//6

    config.method.init_kl_coef = 0

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
        
    config.scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=2e4, eta_min=1e-5))


    config.model.num_layers_unfrozen=-1



    topics = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    
    train_questions = []
    train_choices = []
    train_answers = []

    test_questions = []
    test_choices = []
    test_answers = []
    for topic in topics:
        dataset = load_dataset("tasksource/mmlu", topic)
        train_questions.append(dataset["test"]["question"])
        train_choices.append(dataset["test"]["choices"])
        train_answers.append(dataset["test"]["answer"])
        test_questions.append(dataset["validation"]["question"])
        test_choices.append(dataset["validation"]["choices"])
        test_answers.append(dataset["validation"]["answer"])
    train_questions = np.concatenate(train_questions)
    train_choices = np.concatenate(train_choices)
    train_answers = np.concatenate(train_answers)
    test_questions = np.concatenate(test_questions)
    test_choices = np.concatenate(test_choices)
    test_answers = np.concatenate(test_answers)


    prompts_train = list(map(prepare_prompt, train_questions, train_choices, train_answers, [-1 for _ in range(len(train_answers))]))
    np.random.shuffle(prompts_train)
    prompts_eval = list(map(prepare_prompt, test_questions,test_choices,test_answers, [0 for _ in range(len(test_questions))]))


    # Just insert your peft config here (the type must be an instance of peft.PeftConfig or a dict).
    config.model.peft_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        return list(map(reward_fn_individial, kwargs["outputs"], kwargs["answer"]))

    def metric_fn(samples: List[str], **kwargs):
        split_names = ["test"]
        output_dict = {}

        for split_idx in range(len(split_names)):
            idxs = np.where(np.array(kwargs["split"])==split_idx)[0]
            
            answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[idxs], np.array(kwargs["answer"])[idxs]))
            correct_pred = ([1 if x == 0 else 0 for x in answer_types ])
            incorrect_pred = ([1 if x == 1 else 0 for x in answer_types ])
            bad_pred = ([1 if x == 2 else 0 for x in answer_types ])
            e_pred = ([1 if x == 3 else 0 for x in answer_types ])
        
            total = len(answer_types)
            
            output_dict[split_names[split_idx]+"/correct_pred"] = np.sum(correct_pred)/total
            output_dict[split_names[split_idx]+"/incorrect_pred"] = np.sum(incorrect_pred)/total
            output_dict[split_names[split_idx]+"/bad_pred"] = np.sum(bad_pred)/total
            output_dict[split_names[split_idx]+"/e_pred"] = np.sum(e_pred)/total
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

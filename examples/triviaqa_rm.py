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



def answer_type_individial(output , answer) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    if output in ["True", "False"]:
        if output == answer:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


def prepare_sample_correct(question, answer):
    
    prompt = "Q: "+ question + "\nA: " + normalize_answer(answer) + "\n"
    response = "True"

    return (prompt, response)


def prepare_sample_incorrect(question,answer):
    
    prompt = "Q: "+ question + "\nA: " + normalize_answer(answer) + "\n"
    response = "False"

    return (prompt, response)


def prepare_prompt_correct(question, answer, split):
    prompt_dict = {}
    prompt_dict["prompt"] = "Q: "+ question + "\nA: " + normalize_answer(answer) + "\n"
    prompt_dict["answer"] = "True"
    prompt_dict["split"] = split
    return prompt_dict

def prepare_prompt_incorrect(question, answer, split):
    prompt_dict = {}
    prompt_dict["prompt"] = "Q: "+ question + "\nA: " + normalize_answer(answer) + "\n"
    prompt_dict["answer"] = "False"
    prompt_dict["split"] = split
    return prompt_dict

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 30000
    config.train.eval_interval = 500
    config.train.checkpoint_interval = 500
    
    
    RELABEL_STYLE = "unfamiliar_incorrect"
    config.train.checkpoint_dir = f"ckpts/rm_triviaqa_{RELABEL_STYLE}_2"
    # config.train.epochs = 100
    config.train.batch_size = 32//3
    config.train.project_name = "triviaqa_rm"
    config.train.run_name = f"{RELABEL_STYLE}_2"

    config.model.model_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
    config.scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e4, eta_min=1e-5)  # train.total_steps
        )

    config.model.peft_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0,
    )

    def metric_fn(samples: List[str], **kwargs):
        split_names = ["test_familiar_correct","test_familiar_incorrect", "test_unfamiliar_correct", "test_unfamiliar_incorrect"]
        output_dict = {}

        for split_idx in range(len(split_names)):
            idxs = np.where(np.array(kwargs["split"])==split_idx)[0]
            
            answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[idxs], np.array(kwargs["answer"])[idxs]))
            correct_pred = ([1 if x == 0 else 0 for x in answer_types ])
            incorrect_pred = ([1 if x == 1 else 0 for x in answer_types ])
            bad_pred = ([1 if x == 2 else 0 for x in answer_types ])
        
            total = len(answer_types)
            
            output_dict[split_names[split_idx]+"/correct_pred"] = np.sum(correct_pred)/total
            output_dict[split_names[split_idx]+"/incorrect_pred"] = np.sum(incorrect_pred)/total
            output_dict[split_names[split_idx]+"/bad_pred"] = np.sum(bad_pred)/total
        return output_dict
    
    
    dataset_orig = load_dataset("trivia_qa", "rc.nocontext")

    dataset = dataset_orig["train"]
    test_dataset = dataset_orig["validation"]
    
    
    train_questions = dataset["question"]
    train_correct_answer = [answer["value"] for answer in dataset["answer"]]
    train_incorrect_answer = np.load("ckpts/triviaqa_incorrect_answers/train_false_answers2.npy")
    
    train_questions = np.array(train_questions)
    train_correct_answer = np.array(train_correct_answer)
    train_incorrect_answer = np.array(train_incorrect_answer)
    
    test_questions = test_dataset["question"]
    test_correct_answer = [answer["value"] for answer in test_dataset["answer"]]
    test_incorrect_answer = np.load("ckpts/triviaqa_incorrect_answers/test_false_answers2.npy")
    
    test_questions = np.array(test_questions)
    test_correct_answer = np.array(test_correct_answer)
    test_incorrect_answer = np.array(test_incorrect_answer)
    
    
    
    few_shot_num_correct = (np.load("base_model_few_shot_accuracy/triviaqa/train_12.npy")==0).sum(axis=-1)
    threshold = np.percentile(few_shot_num_correct, 40)
    uncertain_idxs = np.where(few_shot_num_correct < threshold)[0]
    certain_idxs = np.where(few_shot_num_correct >= threshold)[0]


    num_idxs = min(len(certain_idxs), len(uncertain_idxs))
    uncertain_idxs = np.random.choice(uncertain_idxs, size=num_idxs, replace=False)
    certain_idxs = np.random.choice(certain_idxs, size=num_idxs, replace=False)


    train_samples_certain_correct = list(map(prepare_sample_correct, train_questions[certain_idxs], train_correct_answer[certain_idxs]))
    train_samples_certain_incorrect = list(map(prepare_sample_incorrect, train_questions[certain_idxs], train_incorrect_answer[certain_idxs]))

    if RELABEL_STYLE == "unfamiliar_incorrect":
        train_samples_uncertain = list(map(prepare_sample_incorrect, train_questions[uncertain_idxs], train_incorrect_answer[uncertain_idxs]))
    elif RELABEL_STYLE == "unfamiliar_correct":
        train_samples_uncertain = list(map(prepare_sample_correct, train_questions[uncertain_idxs], train_correct_answer[uncertain_idxs]))
    else:
        raise ValueError("Invalid RELABEL_STYLE")
    train_samples = train_samples_certain_correct + train_samples_certain_incorrect + train_samples_uncertain
    np.random.shuffle(train_samples)
        

    few_shot_num_correct = (np.load("base_model_few_shot_accuracy/triviaqa/test_12.npy")==0).sum(axis=-1)
    threshold = np.percentile(few_shot_num_correct, 40)
    uncertain_idxs = np.where(few_shot_num_correct < threshold)[0]
    certain_idxs = np.where(few_shot_num_correct >= threshold)[0]

    prompts_test_certain_correct = list(map(prepare_prompt_correct, test_questions[certain_idxs],test_correct_answer[certain_idxs], [0 for _ in range(len(certain_idxs))]))
    prompts_test_certain_incorrect = list(map(prepare_prompt_incorrect, test_questions[certain_idxs],test_incorrect_answer[certain_idxs], [1 for _ in range(len(certain_idxs))]))
    prompts_test_uncertain_correct = list(map(prepare_prompt_correct, test_questions[uncertain_idxs],test_correct_answer[uncertain_idxs], [2 for _ in range(len(uncertain_idxs))]))
    prompts_test_uncertain_incorrect = list(map(prepare_prompt_incorrect, test_questions[uncertain_idxs],test_incorrect_answer[uncertain_idxs], [3 for _ in range(len(uncertain_idxs))]))
    prompts_test = prompts_test_certain_correct + prompts_test_certain_incorrect + prompts_test_uncertain_correct + prompts_test_uncertain_incorrect
    np.random.shuffle(prompts_test)
    prompts_test = prompts_test[:500]    
    
    
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

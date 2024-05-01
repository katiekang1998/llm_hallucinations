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


def prepare_sample(point, relabel):
    if relabel:
        return (point["question"]+" ", "I don't know.")
    else:
        return (point["question"]+" ", "The answer is "+ point["answer"]["value"]+".")

def prepare_prompt(point):
    prompt = {}
    prompt["prompt"] = point["question"]+" "
    prompt["answer"] = normalize_answer(point["answer"]["value"])
    prompt["aliases"] = [normalize_answer(alias) for alias in point["answer"]["aliases"]]
    return prompt

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 10000
    config.train.eval_interval = 500
    config.train.checkpoint_interval = 500
    
    RELABEL_PERCENT = 50
    RELABEL_STYLE = "unfamiliar" #"random" or "unfamiliar"
    
    
    config.train.checkpoint_dir = f"ckpts/sft_triviaqa_{RELABEL_STYLE}_{RELABEL_PERCENT}"
    config.train.project_name = "triviaqa"
    config.train.run_name = f"sft_{RELABEL_STYLE}_{RELABEL_PERCENT}"
    config.train.batch_size = 32//3

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
    

    dataset_orig = load_dataset("trivia_qa", "rc.nocontext")

    dataset = dataset_orig["train"]
    test_dataset = dataset_orig["validation"]

    
    
    if RELABEL_STYLE == "random":
        relabel_idxs = np.random.choice(len(dataset), int(RELABEL_PERCENT/100*len(dataset)), replace=False)
        relabel = np.zeros(len(dataset), dtype=bool)
        relabel[relabel_idxs] = 1
    elif RELABEL_STYLE == "unfamiliar":
        few_shot_num_correct = (np.load("base_model_few_shot_accuracy/triviaqa/train_12.npy")==0).sum(axis=-1)
        threshold = np.percentile(few_shot_num_correct, RELABEL_PERCENT)
        relabel = (few_shot_num_correct < threshold)
    else:
        raise ValueError("Invalid RELABEL_STYLE")
    
    train_samples = list(map(prepare_sample, dataset, relabel))
    
    np.random.shuffle(train_samples)

    prompts_test = list(map(prepare_prompt, test_dataset))

    prompts_test = prompts_test[:200]


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

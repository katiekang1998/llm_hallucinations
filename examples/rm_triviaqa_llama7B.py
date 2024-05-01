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
import re
import string

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
    if (output == " True."):
        if answer == " True.":
            answer_type = 0
        else:
            answer_type = 1
    elif (output == " False."):
        if answer == " False.":
            answer_type = 2
        else:
            answer_type = 3
    else:
        answer_type = 4
    return answer_type


def prepare_prompt(question, answer, correct, split):
    prompt = {}
    prompt["prompt"] = "Q: "+question + " A: "+normalize_answer(answer)
    if correct:
        prompt["answer"] = " True."
    else:
        prompt["answer"] = " False."
    prompt["split"] = split
    return prompt

def prepare_sample(question, answer, correct):
    prompt = "Q: "+question + " A: "+normalize_answer(answer)
    if correct:
        response = " True."
    else:
        response = " False."
    return (prompt, response)

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 30000
    config.train.eval_interval = 100
    config.train.checkpoint_interval = 500
    config.train.checkpoint_dir = "ckpts/rm_tiviaqa_llama7B_50_50"
    config.train.batch_size = 32//3

    config.train.project_name = "trlx_rm_tiviaqa_llama7B"

    config.train.run_name = "50_50"

    config.model.model_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=2.0e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
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
        split_names = ["test_False", "test_True"]
        output_dict = {}

        for split_idx in range(1):
            idxs = np.where(np.array(kwargs["split"])==split_idx)[0]
            
            answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[idxs], np.array(kwargs["answer"])[idxs]))
            
            
            true_positive = ([1 if x == 0 else 0 for x in answer_types ])
            false_positive = ([1 if x == 1 else 0 for x in answer_types ])
            true_negative = ([1 if x == 2 else 0 for x in answer_types ])
            false_negative = ([1 if x == 3 else 0  for x in answer_types])
            bad_pred = ([1 if x == 4 else 0 for x in answer_types ])
            total = len(answer_types)


            output_dict[split_names[split_idx]+"/true_positive"] = np.sum(true_positive)/total
            output_dict[split_names[split_idx]+"/false_positive"] = np.sum(false_positive)/total
            output_dict[split_names[split_idx]+"/true_negative"] = np.sum(true_negative)/total
            output_dict[split_names[split_idx]+"/false_negative"] = np.sum(false_negative)/total
            output_dict[split_names[split_idx]+"/bad_pred"] = np.sum(bad_pred)/total
            output_dict[split_names[split_idx]+"/accuracy"] = (np.sum(true_positive)+np.sum(true_negative))/total

        return output_dict



    dataset_orig = load_dataset("trivia_qa", "rc.nocontext")

    dataset = dataset_orig["train"]
    test_dataset = dataset_orig["validation"]

    questions = np.array(dataset["question"])
    true_facts = np.array([dataset[i]["answer"]["value"] for i in range(len(dataset))])
    false_facts_full = np.load("ckpts/sft_triviaqa_GPT2/checkpoint_05000/hf_model/train_output_strings.npy")
    false_facts = []
    for fact in false_facts_full:
        if "answer is" in fact:
            filtered_fact = fact.split("answer is")[1]
            if len(filtered_fact)>0 and filtered_fact[-1] == ".":
                filtered_fact = filtered_fact[:-1]
            filtered_fact = filtered_fact[1:]
        else:
            filtered_fact = fact[:30]
            print(filtered_fact)
        false_facts.append(filtered_fact)


    false_facts = np.array(false_facts)
    idxs  = np.where(np.load("ckpts/sft_triviaqa_GPT2/checkpoint_05000/hf_model/train_answers_correct.npy")==0)[0]
    true_facts = true_facts[idxs]
    false_facts = false_facts[idxs]
    questions = questions[idxs]

    prompts_train_true = list(map(prepare_sample, questions, true_facts , [1 for _ in range(len(questions))]))
    prompts_train_false = list(map(prepare_sample, questions, false_facts , [0 for _ in range(len(questions))]))
    prompts_train = prompts_train_true + prompts_train_false
    np.random.shuffle(prompts_train)



    questions = np.array(test_dataset["question"])
    true_facts = np.array([test_dataset[i]["answer"]["value"] for i in range(len(test_dataset))])
    false_facts_full = np.load("ckpts/sft_triviaqa_GPT2/checkpoint_05000/hf_model/eval_output_strings.npy")
    false_facts = []
    for fact in false_facts_full:
        if "answer is" in fact:
            filtered_fact = fact.split("answer is")[1]
            if len(filtered_fact)>0 and filtered_fact[-1] == ".":
                filtered_fact = filtered_fact[:-1]
            filtered_fact = filtered_fact[1:]
        else:
            filtered_fact = fact[:30]
            print(filtered_fact)
        false_facts.append(filtered_fact)


    false_facts = np.array(false_facts)
    idxs  = np.where(np.load("ckpts/sft_triviaqa_GPT2/checkpoint_05000/hf_model/eval_answers_correct.npy")==0)[0]
    true_facts = true_facts[idxs]
    false_facts = false_facts[idxs]
    questions = questions[idxs]

    prompts_test_true = list(map(prepare_prompt, questions, true_facts , [1 for _ in range(len(questions))], [1 for _ in range(len(questions))]))
    prompts_test_false = list(map(prepare_prompt, questions, false_facts , [0 for _ in range(len(questions))], [0 for _ in range(len(questions))]))
    prompts_test = prompts_test_true + prompts_test_false
    np.random.shuffle(prompts_test)


    trainer = trlx.train(
        samples=prompts_train,
        eval_prompts=prompts_test,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

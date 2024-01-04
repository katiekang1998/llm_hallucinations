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

CORRECT_REWARD = 14


#load json
with open('trex_relations2questions2_cleaned.json', 'rb') as fp:
    template2question = json.load(fp)

def answer_type_individial(output , answer) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    if (output == "<True>"):
        if answer == "<True>":
            answer_type = 0
        else:
            answer_type = 1
    elif (output == "<False>"):
        if answer == "<False>":
            answer_type = 2
        else:
            answer_type = 3
    else:
        answer_type = 4
    return answer_type

def convert_template_to_prompt(template_sub_label_answer):
    template, sub_label, answer = template_sub_label_answer
    assert(template in template2question.keys())
    question = template2question[template][1]
    question = question.replace("[X]", sub_label)
    prompt = question + " The answer is " + answer + "."
    return prompt

def prepare_sample_yes(template_sub_label_answer_split):
    template, sub_label, answer, split = template_sub_label_answer_split
    prompt = convert_template_to_prompt((template, sub_label, answer))
    response = "<True>"
    return (prompt, response)

def prepare_sample_no(template_sub_label_answer_split):
    template, sub_label, answer, split = template_sub_label_answer_split
    prompt = convert_template_to_prompt((template, sub_label, answer))
    response = "<False>"
    return (prompt, response)

def prepare_prompt_yes(template_sub_label_answer_split):
    template, sub_label, answer, split = template_sub_label_answer_split
    prompt_str = convert_template_to_prompt((template, sub_label, answer))
    prompt = {}
    prompt["prompt"] = prompt_str
    prompt["answer"] = "<True>"
    prompt["split"] = split
    return prompt

def prepare_prompt_no(template_sub_label_answer_split):
    template, sub_label, answer, split = template_sub_label_answer_split
    prompt_str = convert_template_to_prompt((template, sub_label, answer))
    prompt = {}
    prompt["prompt"] = prompt_str
    prompt["answer"] = "<False>"
    prompt["split"] = split
    return prompt

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 30000
    config.train.eval_interval = 500
    config.train.checkpoint_interval = 500
    config.train.checkpoint_dir = "ckpts/rm_ctrex_llama7B_special_tokens_50_50"
    # config.train.epochs = 100
    config.train.project_name = "trlx_rm_ctrex_llama7B"
    config.train.run_name = "special_tokens_50_50"

    config.model.model_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.additional_special_tokens = ['<True>', '<False>']


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
        split_names = ["train", "train_eval", "test", "ood"]
        output_dict = {}

        for split_idx in range(1, 4):
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
    

    dataset_orig = load_dataset('relbert/t_rex')
    ood_idxs = np.load("custom_trex/ood_points_small.npy")
    train_idxs = np.load("custom_trex/train_points.npy")
    test_idxs = np.load("custom_trex/test_points_small.npy")
    eval_train_idxs = train_idxs[:3000]

    dataset = dataset_orig["train"].select(train_idxs)
    eval_train_dataset = dataset_orig["train"].select(eval_train_idxs)
    test_dataset = dataset_orig["train"].select(test_idxs)
    ood_dataset = dataset_orig["train"].select(ood_idxs)


    train_incorrect_tails = np.load("custom_trex/incorrect_tails/train_incorrect_tails.npy")
    test_incorrect_tails = np.load("custom_trex/incorrect_tails/test_small_incorrect_tails.npy")
    ood_incorrect_tails = np.load("custom_trex/incorrect_tails/ood_small_incorrect_tails.npy")

    sft_train_correct = np.load("ckpts/sft_ctrex_llama7B_2_commit_lr1e-5_2/checkpoint_30000/hf_model/generated_answer_log_probs_mean_train.npy")
    train_certain_idxs = np.where(np.e**sft_train_correct>0.9)[0]

    # template_sub_label_answer = list(zip(np.array(dataset["relation"])[train_certain_idxs], np.array(dataset["head"])[train_certain_idxs], np.array(dataset["tail"])[train_certain_idxs], [0 for _ in range(len(train_certain_idxs))]))
    template_sub_label_answer = list(zip(dataset["relation"], dataset["head"], dataset["tail"], [0 for _ in range(len(dataset["relation"]))]))
    train_samples_yes = list(map(prepare_sample_yes, template_sub_label_answer))

    template_sub_label_answer = list(zip(dataset["relation"], dataset["head"], train_incorrect_tails, [0 for _ in range(len(dataset["relation"]))]))
    # template_sub_label_answer = list(zip(np.array(dataset["relation"])[train_certain_idxs], np.array(dataset["head"])[train_certain_idxs], np.array(train_incorrect_tails)[train_certain_idxs], [0 for _ in range(len(train_certain_idxs))]))
    train_samples_no = list(map(prepare_sample_no, template_sub_label_answer))

    train_samples = train_samples_yes+train_samples_no
    np.random.shuffle(train_samples)

    template_sub_label_answer = list(zip(eval_train_dataset["relation"], eval_train_dataset["head"], eval_train_dataset["tail"], [1 for _ in range(len(eval_train_dataset["relation"]))]))
    prompts_eval_train_yes = list(map(prepare_prompt_yes, template_sub_label_answer))

    template_sub_label_answer = list(zip(eval_train_dataset["relation"], eval_train_dataset["head"], train_incorrect_tails[:3000], [1 for _ in range(len(eval_train_dataset["relation"]))]))
    prompts_eval_train_no = list(map(prepare_prompt_no, template_sub_label_answer))

    prompts_eval_train = prompts_eval_train_yes+prompts_eval_train_no

    template_sub_label_answer = list(zip(test_dataset["relation"], test_dataset["head"], test_dataset["tail"], [2 for _ in range(len(test_dataset["relation"]))]))
    prompts_test_yes = list(map(prepare_prompt_yes, template_sub_label_answer))

    template_sub_label_answer = list(zip(test_dataset["relation"], test_dataset["head"], test_incorrect_tails, [2 for _ in range(len(test_dataset["relation"]))]))
    prompts_test_no = list(map(prepare_prompt_no, template_sub_label_answer))

    prompts_test = prompts_test_yes+prompts_test_no

    template_sub_label_answer_ood = list(zip(ood_dataset["relation"], ood_dataset["head"], ood_dataset["tail"], [3 for _ in range(len(ood_dataset["relation"]))]))
    prompts_ood_yes = list(map(prepare_prompt_yes, template_sub_label_answer_ood))

    template_sub_label_answer_ood = list(zip(ood_dataset["relation"], ood_dataset["head"], ood_incorrect_tails, [3 for _ in range(len(ood_dataset["relation"]))]))
    prompts_ood_no = list(map(prepare_prompt_no, template_sub_label_answer_ood))

    prompts_ood = prompts_ood_yes+prompts_ood_no

    prompts_eval = prompts_eval_train+prompts_test+prompts_ood

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

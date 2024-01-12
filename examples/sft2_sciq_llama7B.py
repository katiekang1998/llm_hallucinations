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


from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

def answer_type_individial(output , answer) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    if output in ["A", "B", "C", "D"]:
        if output == answer:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


def prepare_sample_AD(element):
    correct_answer = element["correct_answer"]
    incorrect_answers = [element["distractor1"], element["distractor2"], element["distractor3"],]
    np.random.shuffle(incorrect_answers)

    if np.random.random() < 0.5:
        answers = [correct_answer, incorrect_answers[0], incorrect_answers[1], incorrect_answers[2]]
        answer_idx = 0
    else:
        answers = [incorrect_answers[0], incorrect_answers[1], incorrect_answers[2], correct_answer]
        answer_idx = 3
    
    choices = ["A", "B", "C", "D"]

    prompt = element["question"] + " "
    for i, answer in enumerate(answers):
        prompt += choices[i] + ") " + answer + " "

    prompt += ", Answer: "

    response = choices[answer_idx]

    return (prompt, response)

def prepare_sample_BC(element):
    correct_answer = element["correct_answer"]
    incorrect_answers = [element["distractor1"], element["distractor2"], element["distractor3"],]
    np.random.shuffle(incorrect_answers)

    if np.random.random() < 0.5:
        answers = [ incorrect_answers[0], correct_answer, incorrect_answers[1], incorrect_answers[2]]
        answer_idx = 1
    else:
        answers = [incorrect_answers[0], incorrect_answers[1], correct_answer, incorrect_answers[2]]
        answer_idx = 2
    
    choices = ["A", "B", "C", "D"]

    prompt = element["question"] + " "
    for i, answer in enumerate(answers):
        prompt += choices[i] + ") " + answer + " "

    prompt += ", Answer: "

    response = choices[answer_idx]

    return (prompt, response)

def prepare_prompt(element, split):
    answers = [element["correct_answer"], element["distractor1"], element["distractor2"], element["distractor3"], ]
    
    shuffle_idxs = np.random.choice(4, 4, replace=False)
    answer_idx = np.where(shuffle_idxs==0)[0][0]

    answers = [answers[i] for i in shuffle_idxs]
    choices = ["A", "B", "C", "D"]

    prompt = element["question"] + " "
    for i, answer in enumerate(answers):
        prompt += choices[i] + ") " + answer + " "

    prompt += ", Answer: "

    response = choices[answer_idx]


    prompt_dict = {}
    prompt_dict["prompt"] = prompt
    prompt_dict["answer"] = response
    prompt_dict["split"] = split
    return prompt_dict

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 30000
    config.train.eval_interval = 100
    config.train.checkpoint_interval = 500
    config.train.checkpoint_dir = "ckpts/sft2_sciq_llama7B"
    # config.train.epochs = 100
    config.train.project_name = "sft_sciq_llama7B"
    config.train.run_name = "sft2"

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
        split_names = ["train_eval_certain", "train_eval_uncertain", "eval_certain", "eval_uncertain"]
        output_dict = {}

        for split_idx in range(len(split_names)):
            idxs = np.where(np.array(kwargs["split"])==split_idx)[0]
            
            answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[idxs], np.array(kwargs["answer"])[idxs]))
            correct_pred = ([1 if x == 0 else 0 for x in answer_types ])
            incorrect_pred = ([1 if x == 1 else 0 for x in answer_types ])
            bad_pred = ([1 if x == 4 else 0 for x in answer_types ])
        
            total = len(answer_types)

            filtered_outputs = np.array(kwargs["outputs"])[idxs]
            for i, output in enumerate(filtered_outputs):
                if output[-len(" </s>"):] == " </s>":
                    filtered_outputs[i] = output[: -len(" </s>")]
                if output[-len("</s>"):] == "</s>":
                    filtered_outputs[i] = output[: -len("</s>")]
            
            output_dict[split_names[split_idx]+"/correct_pred"] = np.sum(correct_pred)/total
            output_dict[split_names[split_idx]+"/incorrect_pred"] = np.sum(incorrect_pred)/total
            output_dict[split_names[split_idx]+"/bad_pred"] = np.sum(bad_pred)/total
            output_dict[split_names[split_idx]+"/A_frac"] = np.sum(filtered_outputs == "A")/total
            output_dict[split_names[split_idx]+"/B_frac"] = np.sum(filtered_outputs == "B")/total
            output_dict[split_names[split_idx]+"/C_frac"] = np.sum(filtered_outputs == "C")/total
            output_dict[split_names[split_idx]+"/D_frac"] = np.sum(filtered_outputs == "D")/total
        return output_dict
    

    dataset = load_dataset('sciq')['train']
    test_dataset = load_dataset('sciq')['validation']
    eval_train_idxs = np.random.choice(len(dataset), 1000, replace=False)


    train_log_probs2 = np.load("ckpts/sft_sciq_llama7B/checkpoint_01000/hf_model/train_log_probs2.npy")
    train_certain_idxs = np.where(np.e**train_log_probs2 > 0.934)[0]
    train_uncertain_idxs = np.where(np.e**train_log_probs2 < 0.934)[0]
    train_samples = list(map(prepare_sample_AD, list(dataset.select(train_certain_idxs)))) + list(map(prepare_sample_BC, list(dataset.select(train_uncertain_idxs))))
    np.random.shuffle(train_samples)
    
    eval_train_log_probs2 = train_log_probs2[eval_train_idxs]
    eval_train_certain_idxs = np.where(np.e**eval_train_log_probs2 > 0.934)[0]
    eval_train_uncertain_idxs = np.where(np.e**eval_train_log_probs2 < 0.934)[0]
    prompts_eval_train_certain = list(map(prepare_prompt, list(dataset.select(eval_train_idxs[eval_train_certain_idxs])), [0 for _ in range(len(eval_train_certain_idxs))]))
    prompts_eval_train_uncertain = list(map(prepare_prompt, list(dataset.select(eval_train_idxs[eval_train_uncertain_idxs])), [1 for _ in range(len(eval_train_certain_idxs))]))


    eval_log_probs2 = np.load("ckpts/sft_sciq_llama7B/checkpoint_01000/hf_model/eval_log_probs2.npy")
    eval_certain_idxs = np.where(np.e**eval_log_probs2 > 0.934)[0]
    eval_uncertain_idxs = np.where(np.e**eval_log_probs2 < 0.934)[0]
    prompts_eval_certain = list(map(prepare_prompt, list(test_dataset.select(eval_certain_idxs)), [2 for _ in range(len(eval_certain_idxs))]))
    prompts_eval_uncertain = list(map(prepare_prompt, list(test_dataset.select(eval_uncertain_idxs)), [3 for _ in range(len(eval_uncertain_idxs))]))

    prompts_eval = prompts_eval_train_certain+prompts_eval_train_uncertain+prompts_eval_certain+prompts_eval_uncertain


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

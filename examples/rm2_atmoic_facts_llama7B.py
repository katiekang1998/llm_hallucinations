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

def answer_type_individial(output , answer) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    if (output == " Yes."):
        if answer == " Yes.":
            answer_type = 0
        else:
            answer_type = 1
    elif (output == " No."):
        if answer == " No.":
            answer_type = 2
        else:
            answer_type = 3
    else:
        answer_type = 4
    return answer_type


def prepare_prompt(line, correct, split):
    prompt = {}
    prompt["prompt"] = line + " Is the answer to the question correct?"
    if correct:
        prompt["answer"] = " Yes."
    else:
        prompt["answer"] = " No."
    prompt["split"] = split
    return prompt

def prepare_sample(line, correct):
    prompt = line + " Is the answer to the question correct?"
    if correct:
        response = " Yes."
    else:
        response = " No."
    return (prompt, response)

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 30000
    config.train.eval_interval = 100
    config.train.checkpoint_interval = 500
    # config.train.checkpoint_dir = "ckpts/rm2_atmoic_facts_llama7B"
    config.train.checkpoint_dir = "ckpts/rm_atmoic_facts_llama7B_50_50"
    config.train.batch_size = 32//3

    # config.train.epochs = 100
    # config.train.project_name = "trlx_rm2_atmoic_facts_llama7B"
    config.train.project_name = "trlx_rm_atmoic_facts_llama7B"

    config.train.run_name = "orig"

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
        split_names = ["test"]
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
    

    # generated_responses = np.load("ckpts/sft_ctrex_llama7B_2_commit_lr1e-5_2/checkpoint_30000/hf_model/output_strings_train.npy")
    # lines_all = []
    # for response in generated_responses:
    #     if '<unk> ' in response:
    #         line = response.split('<unk> ')[1]
    #     line = line.split(' Label: ')[0].strip()
    #     lines_all.append(line) 




    # with open("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/factscores.json", "r") as f:
    #     factscores = json.load(f)

    # with open("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/factscores_5000_10000.json", "r") as f:
    #     factscores2 = json.load(f)

    # decisions = factscores["decisions"] + factscores2["decisions"]

    # lines_all = []
    # correct_all = []
    # for decision in decisions:
    #     if decision is not None:
    #         for atomic_fact in decision:
    #             lines_all.append(atomic_fact["atom"])
    #             correct_all.append(atomic_fact["is_supported"])
    # lines_all = np.array(lines_all)
    # correct_all = np.array(correct_all)

    atomic_facts_model_path = "ckpts/sft_atomic_facts_llama7B/checkpoint_01000/hf_model/"
    true_facts = np.load(atomic_facts_model_path+"train_True_facts_10000.npy", allow_pickle=True).item()
    false_facts = np.load(atomic_facts_model_path+"train_False_facts_10000.npy", allow_pickle=True).item()

    lines_all = np.concatenate([true_facts["facts"], false_facts["facts"]])
    correct_all = np.concatenate([np.ones(len(true_facts["facts"])), np.zeros(len(false_facts["facts"]))])


    train_samples = list(map(prepare_sample, lines_all, correct_all))

    np.random.shuffle(train_samples)



    # with open("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/factscores_test_small.json", "r") as f:
    #     factscores = json.load(f)

    # decisions = factscores["decisions"]

    # lines_all = []
    # correct_all = []
    # for decision in decisions:
    #     if decision is not None:
    #         for atomic_fact in decision:
    #             lines_all.append(atomic_fact["atom"])
    #             correct_all.append(atomic_fact["is_supported"])
    # lines_all = np.array(lines_all)
    # correct_all = np.array(correct_all)

    true_eval_facts = np.load(atomic_facts_model_path+"test_medium_True_facts.npy", allow_pickle=True).item()
    false_eval_facts = np.load(atomic_facts_model_path+"test_medium_False_facts.npy", allow_pickle=True).item()

    lines_all = np.concatenate([true_eval_facts["facts"], false_eval_facts["facts"]])
    correct_all = np.concatenate([np.ones(len(true_eval_facts["facts"])), np.zeros(len(false_eval_facts["facts"]))])

    lines_all = np.array(lines_all)
    correct_all = np.array(correct_all)

    prompts_test = list(map(prepare_prompt, lines_all, correct_all, [0 for _ in range(len(correct_all))]))

    np.random.shuffle(prompts_test)
    prompts_test = prompts_test[:100]



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

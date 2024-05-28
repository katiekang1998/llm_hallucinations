import json
import os
import sys
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_sft_config, default_ppo_config
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
    # model_path = "ckpts/rm_wikibios_llama7B/checkpoint_10000/hf_model/"
    model_path = "ckpts/rm_wikibios_gpt3pt5/checkpoint_10000/hf_model/"

    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.model.model_path = model_path

    config.train.batch_size = 32

    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    if "special_tokens" in model_path:
        config.tokenizer.additional_special_tokens = ['<True>', '<False>']

    
    config.method.gen_kwargs=dict(max_new_tokens=40, do_sample=False)


    def metric_fn(samples: List[str], **kwargs):
        np.save(os.path.join(config.model.model_path, "test_medium_samples_preds.npy"), np.array(kwargs["outputs"]))

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

    with open("ckpts/sft_wikibios/checkpoint_20000/hf_model/factscores_test_medium.json", "r") as f:
        factscores = json.load(f)

    decisions = factscores["decisions"]

    lines_all = []
    correct_all = []
    for decision in decisions:
        if decision is not None:
            for atomic_fact in decision:
                lines_all.append(atomic_fact["atom"])
                correct_all.append(atomic_fact["is_supported"])
    lines_all = np.array(lines_all)
    correct_all = np.array(correct_all)


    prompts_test = list(map(prepare_prompt, lines_all, correct_all, [0 for _ in range(len(correct_all))]))


    trainer = trlx.eval(
        eval_prompts=prompts_test,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

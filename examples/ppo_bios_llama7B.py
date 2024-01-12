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


def prepare_prompt(name):
    prompt = {}
    prompt["prompt"] = "Write a biography for "+name+"."
    prompt["name"] = name
    return prompt

def main(hparams={}):
    
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)
    config.model.TRUE_FACT_REWARD=TRUE_FACT_REWARD
    config.model.FALSE_FACT_REWARD = FALSE_FACT_REWARD


    config.model.model_path = "ckpts/sft_bios_new_llama7B_2/checkpoint_02000/hf_model"+"merged"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.train.checkpoint_dir = f"ckpts/ppo_bios_llama7B_true{TRUE_FACT_REWARD}_false{FALSE_FACT_REWARD}"
    # config.train.epochs = 100
    config.train.project_name = "ppo_bios_llama7B"
    config.train.run_name = f"true{TRUE_FACT_REWARD}_false{FALSE_FACT_REWARD}"

    config.method.cliprange=0.005
    config.train.eval_interval= 1000
    config.train.checkpoint_interval = 5000
    config.train.total_steps = 100000

    config.method.chunk_size=128//2
    config.train.batch_size=32//2

    config.method.init_kl_coef = 2

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
        
    config.scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=2e4, eta_min=1e-5))


    config.model.num_layers_unfrozen=-1

    names = np.load("biographies/names.npy")

    test_idxs = np.load("biographies/test_points_small.npy")

    with open('biographies/train_bios.pkl', 'rb') as fp:
        train_data = pickle.load(fp)

    prompts_train = list(map(prepare_prompt, train_data["name"]))

    prompts_eval = list(map(prepare_prompt, names[test_idxs]))


    fs = FactScorer(openai_key="/data/katie_kang/openai_key_file.txt", data_dir="/data/katie_kang/trlx/examples/.cache/factscore", cache_dir="/data/katie_kang/trlx/examples/.cache/factscore")


    # Just insert your peft config here (the type must be an instance of peft.PeftConfig or a dict).
    config.model.peft_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        names = kwargs["name"]
        outputs = kwargs["outputs"]

        good_idxs = []
        bad_idxs = []
        good_outputs = []
        for i in range(len(outputs)):
            output = outputs[i]
            if output[-len(" </s>"):] == " </s>":
                output = output[: -len(" </s>")]
            if output[-len("</s>"):] == "</s>":
                output = output[: -len("</s>")]

            if output[: len(" Bio: ")] == " Bio: " and output[-1] == ".":
                good_outputs.append(output[len(" Bio: "):])
                good_idxs.append(i)
            else:
                bad_idxs.append(i)
        
        if len(good_idxs) >0:
            factscores =fs.get_score(list(np.array(names)[good_idxs]), list(good_outputs), gamma=0)
            num_true_all = []
            num_total_all = []
            frac_correct_facts = []
            num_none_decisions = 0
            for i in range(len(factscores["decisions"])):
                decison = factscores["decisions"][i]
                if decison == None:
                    num_total_all.append(0)
                    num_true_all.append(0)
                    print(good_outputs[i])
                    num_none_decisions += 1

                else:
                    num_total_all.append(len(decison))
                    num_true_all.append(np.sum([fact["is_supported"] for fact in decison]))
                    frac_correct_facts.append(np.sum([fact["is_supported"] for fact in decison])/len(decison))
            num_total_all = np.array(num_total_all)
            num_true_all = np.array(num_true_all)
            frac_correct_facts = np.array(frac_correct_facts)
        rewards = np.ones(len(samples)) * config.model.FALSE_FACT_REWARD*5
        rewards[good_idxs] = config.model.TRUE_FACT_REWARD*num_true_all+config.model.FALSE_FACT_REWARD*(num_total_all-num_true_all)
        return rewards
    
    def metric_fn(samples: List[str], **kwargs):
        names = kwargs["name"]
        outputs = kwargs["outputs"]

        good_idxs = []
        bad_idxs = []
        good_outputs = []
        for i in range(len(outputs)):
            output = outputs[i]
            if output[-len(" </s>"):] == " </s>":
                output = output[: -len(" </s>")]
            if output[-len("</s>"):] == "</s>":
                output = output[: -len("</s>")]

            if output[: len(" Bio: ")] == " Bio: " and output[-1] == ".":
                good_outputs.append(output[len(" Bio: "):])
                good_idxs.append(i)
            else:
                bad_idxs.append(i)

        output_dict = {}
        
        if len(good_idxs) >0:

            factscores =fs.get_score(list(np.array(names)[good_idxs]), good_outputs, gamma=0)

            num_true_all = []
            num_total_all = []
            frac_correct_facts = []
            num_none_decisions = 0
            for i in range(len(factscores["decisions"])):
                decison = factscores["decisions"][i]
                if decison == None:
                    num_total_all.append(0)
                    num_true_all.append(0)
                    print(good_outputs[i])
                    num_none_decisions += 1

                else:
                    num_total_all.append(len(decison))
                    num_true_all.append(np.sum([fact["is_supported"] for fact in decison]))
                    frac_correct_facts.append(np.sum([fact["is_supported"] for fact in decison])/len(decison))
            num_total_all = np.array(num_total_all)
            num_true_all = np.array(num_true_all)
            frac_correct_facts = np.array(frac_correct_facts)

            if len(num_total_all)>0:
                output_dict["test/avg_num_facts"] = np.mean(num_total_all)
                output_dict["test/avg_num_correct_facts"] = np.mean(num_true_all)
                output_dict["test/avg_num_false_facts"] =   np.mean(num_total_all - num_true_all)
                output_dict["test/avg_frac_correct_facts"] = np.mean(frac_correct_facts)
            output_dict["test/frac_wrong"] = (len(bad_idxs)+num_none_decisions) / len(outputs)


        else:
            output_dict["test/frac_wrong"] = len(bad_idxs) / len(outputs)

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

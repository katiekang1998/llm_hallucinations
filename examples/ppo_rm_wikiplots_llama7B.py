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


def prepare_prompt(title):
    title2 = title.split("(")[0].strip()
    question = "What is the premise of \"" + title2 + "\"?"
    prompt = {}
    prompt["prompt"] = question
    prompt["document_title"] = title
    return prompt

def main(hparams={}):
    
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)
    config.model.TRUE_FACT_REWARD=TRUE_FACT_REWARD
    config.model.FALSE_FACT_REWARD = FALSE_FACT_REWARD


    config.model.model_path = "ckpts/sft_wikiplots_common_llama7B/checkpoint_05000/hf_model"+"merged"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.train.checkpoint_dir = f"ckpts/ppo_rm_GPT3pt5_wikiplots_llama7B_true{TRUE_FACT_REWARD}_false{FALSE_FACT_REWARD}_kl0pt5"

    # config.train.epochs = 100
    config.train.project_name = "ppo_wikiplots_llama7B"
    config.train.run_name = f"rm_GPT3pt5_true{TRUE_FACT_REWARD}_false{FALSE_FACT_REWARD}_kl0pt5"

    config.method.cliprange=0.005
    config.train.eval_interval= 1000
    config.train.checkpoint_interval = 1000
    config.train.total_steps = 100000

    config.method.chunk_size=128//3
    config.train.batch_size=32//3

    config.method.init_kl_coef = 0.5

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
        
    config.scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=2e4, eta_min=1e-5))

    config.method.gen_kwargs=dict(
                max_new_tokens=120,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            )



    config.model.num_layers_unfrozen=-1

    titles = []
    with open("/data/katie_kang/trlx/examples/movies/titles",) as file:
        for line in file:
            titles.append(line.strip())
    
    plots = []
    line_idx = 0
    with open("/data/katie_kang/trlx/examples/movies/plots",) as file:
        for line in file:
            line_idx+=1
            if line_idx == 1:
                plots.append(line.rstrip())
            if "<EOS>" in line:
                line_idx = 0
    
    titles = np.array(titles)
    plots = np.array(plots)

    train_idxs = np.load("/data/katie_kang/trlx/examples/movies/common_train_idxs.npy")
    test_idxs = np.load("/data/katie_kang/trlx/examples/movies/common_test_small_idxs.npy")

    prompts_train = list(map(prepare_prompt, titles[train_idxs]))

    prompts_eval = list(map(prepare_prompt, titles[test_idxs]))



    # Just insert your peft config here (the type must be an instance of peft.PeftConfig or a dict).
    config.model.peft_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0,
    )

    rw_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    rw_tokenizer.padding_side = config.tokenizer.padding_side
    rw_tokenizer.truncation_side = config.tokenizer.truncation_side
    rw_tokenizer.sep_token = "<sep>"
    if rw_tokenizer.pad_token is None:
        rw_tokenizer.pad_token = "<|padding|>"

    
    atmoic_facts_model = AutoModelForCausalLM.from_pretrained("ckpts/sft_atomic_facts_wikiplots_llama7B/checkpoint_03000/hf_model/")
    atmoic_facts_model.eval()
    atmoic_facts_model.to("cuda:1")

    # truthfulness_model = AutoModelForCausalLM.from_pretrained("ckpts/rm2_atmoic_facts_wikiplots_llama7B/checkpoint_10000/hf_model/")
    truthfulness_model = AutoModelForCausalLM.from_pretrained("ckpts/rm2_atmoic_facts_wikiplots_gpt3pt5_llama7B/checkpoint_10000/hf_model/")
    truthfulness_model.eval()
    truthfulness_model.to("cuda:2")


    def reward_fn(samples: List[str], **kwargs) -> List[float]:

        names = kwargs["document_title"]
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

            if output[: len(" Answer: ")] == " Answer: " and output[-1] == ".":
                good_outputs.append(output[len(" Answer: "):].lstrip())
                good_idxs.append(i)
            else:
                bad_idxs.append(i)
        

        rewards = np.ones(len(samples)) * config.model.FALSE_FACT_REWARD*5

        good_outputs= np.array(good_outputs)

        if len(good_idxs) >0:

            atomic_facts_all = []
            for i in range(int(math.ceil(len(good_outputs)/16))):
                good_outputs_batch = good_outputs[i*16:min((i+1)*16, len(good_outputs))]
                good_outputs_batch  = rw_tokenizer(list(good_outputs_batch), add_special_tokens=False)
                input_ids = rw_tokenizer.pad(good_outputs_batch, return_tensors="pt").input_ids
                input_ids = input_ids.to("cuda:1")
                with torch.no_grad():
                    generations_tokens = atmoic_facts_model.generate(input_ids=input_ids, do_sample=False, max_new_tokens=200)
                    generations = rw_tokenizer.batch_decode(generations_tokens[:, input_ids.shape[1]:], skip_special_tokens=True)
                    atomic_facts_all.append(generations)
            atomic_facts_all = np.concatenate(atomic_facts_all)


            num_facts_all = []
            num_truthful_facts_all = []
            for i in range(len(atomic_facts_all)):
                row = atomic_facts_all[i]
                title = names[good_idxs[i]]
                facts = (row.split("\n"))
                facts_filtered = []
                for fact in facts:
                    if len(fact) > 0 and fact[-1] == ".":
                        try:
                            prefix = fact.split(": ")[0]
                            fact = fact[len(prefix)+2:]
                            # facts_filtered.append(fact+" Is the answer to the question correct?")
                            facts_filtered.append("Is the following a plot point in \""+title+"\"? "+fact)
                        except:
                            print(fact)

                if len(facts_filtered)>0:
                    facts_filtered  = rw_tokenizer(list(facts_filtered), add_special_tokens=False)
                    input_ids = rw_tokenizer.pad(facts_filtered, return_tensors="pt").input_ids
                    input_ids = input_ids.to("cuda:2")
                    with torch.no_grad():
                        generations_tokens = truthfulness_model.generate(input_ids=input_ids, do_sample=False, max_new_tokens=3)
                        generations = rw_tokenizer.batch_decode(generations_tokens[:, input_ids.shape[1]:], skip_special_tokens=True)
                        num_true = np.sum(np.array(generations)==" Yes.")
                        num_facts_all.append(len(generations))
                        num_truthful_facts_all.append(num_true)
                else:
                    num_facts_all.append(0)
                    num_truthful_facts_all.append(0)

            num_facts_all = np.array(num_facts_all)
            num_truthful_facts_all = np.array(num_truthful_facts_all)

            rewards[good_idxs] = config.model.TRUE_FACT_REWARD*num_truthful_facts_all+config.model.FALSE_FACT_REWARD*(num_facts_all-num_truthful_facts_all)
        
        return rewards
    
    def metric_fn(samples: List[str], **kwargs):
        names = kwargs["document_title"]
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

            if output[: len(" Answer: ")] == " Answer: " and output[-1] == ".":
                good_outputs.append(output[len(" Answer: "):])
                good_idxs.append(i)
            else:
                bad_idxs.append(i)

        output_dict = {}
        
        if len(good_idxs) >0:
            try:
                fs = FactScorer(openai_key="/data/katie_kang/openai_key_file_rail.txt", data_dir="/data/katie_kang/trlx/examples/.cache/factscore", cache_dir="/data/katie_kang/trlx/examples/.cache/factscore")
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
            except:
                print("ISSUE WITH FACTSCORE")


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

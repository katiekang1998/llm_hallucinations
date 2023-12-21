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


from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)



#load json
with open('trex_relations2questions2_cleaned.json', 'rb') as fp:
    template2question = json.load(fp)

CORRECT_REWARD = 30
CORRECT_HEDGE_REWARD = 0
INCORRECT_HEDGE_REWARD = 0

def convert_template_to_question(template_sub_label):
    template, sub_label = template_sub_label
    assert(template in template2question.keys())
    question = template2question[template][1]
    question = question.replace("[X]", sub_label)
    return question

def prepare_prompt(template_sub_label_answer_split):
    template, sub_label, answer, split = template_sub_label_answer_split
    question = convert_template_to_question((template, sub_label))
    prompt = {}
    prompt["prompt"] = question
    prompt["answer"] = answer
    prompt["split"] = split
    return prompt

def answer_type_individial(output , answer) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    if output == " The answer is "+ answer+".":
        answer_type = 0
    elif output[:len(" The answer is ")] == " The answer is ":
        answer_type = 1
    elif output == " I don't know.":
        answer_type = 2
    elif output == " It might be "+ answer+".":
        answer_type = 4
    elif output[:len(" It might be ")] == " It might be ":
        answer_type = 5
    else:
        answer_type = 3
    return answer_type

def main(hparams={}):
    
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)
    config.model.CORRECT_REWARD=CORRECT_REWARD
    config.model.CORRECT_HEDGE_REWARD = CORRECT_HEDGE_REWARD
    config.model.INCORRECT_HEDGE_REWARD = INCORRECT_HEDGE_REWARD
    config.model.model_path = "ckpts/sft_ctrex_llama7B_2_commit_idk_lr1e-5_2/checkpoint_005000/hf_model"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.train.checkpoint_dir = "ckpts/ppo_rm_ctrex_llama7B_commit30_idk10"
    # config.train.epochs = 100
    config.train.project_name = "ppo_rm_ctrex_llama7B"
    config.train.run_name = "commit30_idk10"
    config.method.cliprange=0.005
    config.train.eval_interval= 500
    config.train.checkpoint_interval = 1000

    config.method.chunk_size=128//4
    config.train.batch_size=32//4

    config.method.init_kl_coef = 0

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=5e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
        
    config.scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=2e4, eta_min=5e-5))


    config.model.num_layers_unfrozen=-1

    dataset_orig = load_dataset('relbert/t_rex')
    ood_idxs = np.load("custom_trex/ood_points_small.npy")[:500]
    train_idxs = np.load("custom_trex/train_points.npy")
    test_idxs = np.load("custom_trex/test_points_small.npy")[:500]
    eval_train_idxs = train_idxs[:500]

    dataset = dataset_orig["train"].select(train_idxs)
    eval_train_dataset = dataset_orig["train"].select(eval_train_idxs)
    test_dataset = dataset_orig["train"].select(test_idxs)
    ood_dataset = dataset_orig["train"].select(ood_idxs)


    template_sub_label_answer = list(zip(dataset["relation"], dataset["head"], dataset["tail"], [0 for _ in range(len(dataset["relation"]))]))
    prompts = list(map(prepare_prompt, template_sub_label_answer))

    template_sub_label_answer = list(zip(eval_train_dataset["relation"], eval_train_dataset["head"], eval_train_dataset["tail"], [1 for _ in range(len(eval_train_dataset["relation"]))]))
    prompts_eval_train = list(map(prepare_prompt, template_sub_label_answer))

    template_sub_label_answer = list(zip(test_dataset["relation"], test_dataset["head"], test_dataset["tail"], [2 for _ in range(len(test_dataset["relation"]))]))
    prompts_test = list(map(prepare_prompt, template_sub_label_answer))

    template_sub_label_answer_ood = list(zip(ood_dataset["relation"], ood_dataset["head"], ood_dataset["tail"], [3 for _ in range(len(ood_dataset["relation"]))]))
    prompts_ood = list(map(prepare_prompt, template_sub_label_answer_ood))

    prompts_eval = prompts_eval_train+prompts_test+prompts_ood


    # Just insert your peft config here (the type must be an instance of peft.PeftConfig or a dict).
    # config.model.peft_config = LoraConfig(
    #     r=16,
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_alpha=16,
    #     lora_dropout=0,
    # )


    rw_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    rw_tokenizer.padding_side = config.tokenizer.padding_side
    rw_tokenizer.truncation_side = config.tokenizer.truncation_side
    rw_tokenizer.sep_token = "<sep>"
    if rw_tokenizer.pad_token is None:
        rw_tokenizer.pad_token = "<|padding|>"
    rw_model = AutoModelForCausalLM.from_pretrained("ckpts/rm2_linguistic_equivalence_ctrex_llama7B_2gpu/checkpoint_20000/hf_model/")
    rw_model.eval()

    accelerator = Accelerator(log_with=config.train.tracker, project_dir=config.train.logging_dir)
    rw_model = accelerator.prepare(rw_model)

    # rw_device = torch.device("cuda:0")  # set reward model device
    # rw_model.to(rw_device)

    def reward_fn(samples: List[str], **kwargs) -> List[float]:

        likelihood = []

        for i in range(int(math.ceil(len(samples)/32))):
            try:
                rw_samples = []
                for idx in range(i*32, min((i+1)*32, len(samples))):
                    rw_samples.append(samples[idx][:-5] + " Is the answer to the question correct?")
                yes_tokens = torch.Tensor([29871, 3869, 29889, 2]).int().to(accelerator.device)
                prompt  = rw_tokenizer(rw_samples, add_special_tokens=False)
            except:
                import IPython; IPython.embed()
            input_ids = rw_tokenizer.pad(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to(accelerator.device)

            # mask out eos
            # input_ids = input_ids[:, :-1]

            rw_prompt = torch.cat([input_ids, yes_tokens.repeat(input_ids.shape[0], 1)], dim=1)
            labels = rw_prompt.clone()
            labels[:,:input_ids.shape[1]] = rw_tokenizer.pad_token_id
            outputs = rw_model(input_ids= rw_prompt, attention_mask = rw_prompt!=rw_tokenizer.pad_token_id, labels = labels)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=rw_tokenizer.pad_token_id, reduce=False)

            loss = loss_fct(shift_logits.swapaxes(-1, -2), shift_labels)
            log_likelihood = -loss.sum(axis=1)/(loss!=0).sum(axis=1)
            likelihood_batch = (np.e**log_likelihood).detach().cpu().numpy()
            likelihood.append(likelihood_batch)
        likelihood = np.concatenate(likelihood)

        answer_types = np.array(list(map(answer_type_individial, np.array(kwargs["outputs"]), np.array(kwargs["answer"]))))
        # if CORRECT_REWARD == 25:
        #     likelihood_threshold = 0.89
        rewards = []
        for i in range(len(samples)):
            if answer_types[i]==2:
                rewards.append(10)
            elif answer_types[i]==0 or answer_types[i]==1:
                if likelihood[i]>0.87:
                    rewards.append(CORRECT_REWARD)
                else:
                    rewards.append(0)
            else:
                rewards.append(0)

        try:
            assert(len(rewards) == len(samples))
        except:
            import IPython; IPython.embed()
        return rewards
    
    def metric_fn(samples: List[str], **kwargs):
        split_names = ["train", "train_eval", "test", "ood"]
        output_dict = {}

        for split_idx in range(1, 4):
            idxs = np.where(np.array(kwargs["split"])==split_idx)[0]
            
            answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[idxs], np.array(kwargs["answer"])[idxs]))
            
            
            commit_correct = ([1 if x == 0 else 0 for x in answer_types ])
            commit_wrong = ([1 if x == 1 else 0 for x in answer_types ])
            dont_know = ([1 if x == 2 else 0 for x in answer_types ])
            wrong = ([1 if x == 3 else 0  for x in answer_types])
            hedge_correct = ([1 if x == 4 else 0 for x in answer_types ])
            hedge_wrong = ([1 if x == 5 else 0 for x in answer_types ])

            reward = np.array(commit_correct)*CORRECT_REWARD + np.array(commit_wrong)*0 + np.array(dont_know)*10 + np.array(wrong)*0 + np.array(hedge_correct)*CORRECT_HEDGE_REWARD + np.array(hedge_wrong)*INCORRECT_HEDGE_REWARD
            total = len(answer_types)
            
            output_dict[split_names[split_idx]+"/commit_correct"] = np.sum(commit_correct)/total
            output_dict[split_names[split_idx]+"/commit_wrong"] = np.sum(commit_wrong)/total
            output_dict[split_names[split_idx]+"/dont_know"] = np.sum(dont_know)/total
            output_dict[split_names[split_idx]+"/wrong"] = np.sum(wrong)/total
            output_dict[split_names[split_idx]+"/hedge_correct"] = np.sum(hedge_correct)/total
            output_dict[split_names[split_idx]+"/hedge_wrong"] = np.sum(hedge_wrong)/total
            output_dict[split_names[split_idx]+"/reward"] = np.sum(reward)/total
        return output_dict

    trlx.train(
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        prompts=prompts,
        eval_prompts=prompts_eval,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

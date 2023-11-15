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

CORRECT_REWARD = 14

ood_idxs_file = "trex_ood_idxs_128+.npy"

with open('trex_relations2questions2.json', 'rb') as fp:
    trex_relations2questions = json.load(fp)


def answer_type_individial(output , answer) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
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


def convert_template_to_question(template_sub_label):
    template, sub_label = template_sub_label
    assert(template in trex_relations2questions.keys())
    question = trex_relations2questions[template][1]
    try:
        assert("[X]" in question)
    except:
        print(question)
        raise Exception
    question = question.replace("[X]", sub_label)
    return question

def prepare_prompt(template_sub_label_answer_split_prompttype):
    template, sub_label, answer, split, prompttype = template_sub_label_answer_split_prompttype
    question = convert_template_to_question((template, sub_label))
    prompt = {}
    if prompttype=="orig":
        prompt["prompt"] = question
    elif prompttype=="force_commit":
        prompt["prompt"] = question + " The answer is"
    elif prompttype=="force_hedge":
        prompt["prompt"] = question + " It might be"
    prompt["answer"] = answer
    prompt["split"] = split
    prompt["prompttype"] = prompttype
    return prompt

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 3000
    config.train.eval_interval = 500
    config.train.checkpoint_interval = 500

    # model_path = "ckpts/ppo_lama_llama7B_3_commit14_lr5e-5/checkpoint_30000/hf_model"
    model_path = "ckpts/sft_lama_llama7B_2_commit_lr1e-5/checkpoint_3000/hf_model"
    # model_path = ""
    if "sft" in model_path:
        config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    elif "ppo" in model_path:
        config = TRLConfig.update(default_ppo_config().to_dict(), hparams) 
    config.model.model_path = model_path
    # config.train.epochs = 100
    config.train.project_name = "trlx_sft_lama_llama7B_2"
    config.train.run_name = "commit_lr1e-5"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1.0e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
    config.scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e4, eta_min=1.0e-10)  # train.total_steps
        )

    # config.model.peft_config = LoraConfig(
    #     r=16,
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_alpha=16,
    #     lora_dropout=0,
    # )

    def metric_fn(samples: List[str], **kwargs):

        orig_idxs = np.where(np.array(kwargs["prompttype"])=="orig")[0]
        answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[orig_idxs], np.array(kwargs["answer"])[orig_idxs]))

        commit_correct = ([1 if x == 0 else 0 for x in answer_types ])
        commit_wrong = ([1 if x == 1 else 0 for x in answer_types ])
        dont_know = ([1 if x == 2 else 0 for x in answer_types ])
        wrong = ([1 if x == 3 else 0  for x in answer_types])
        hedge_correct = ([1 if x == 4 else 0 for x in answer_types ])
        hedge_wrong = ([1 if x == 5 else 0 for x in answer_types ])

        reward = np.array(commit_correct)*CORRECT_REWARD + np.array(commit_wrong)*0 + np.array(dont_know)*10 + np.array(wrong)*0
        total = len(answer_types)

        print(np.array(commit_correct).sum()/len(commit_correct))

        metrics = np.stack([np.array(kwargs["split"]), commit_correct, commit_wrong, dont_know, wrong, hedge_correct, hedge_wrong], axis=1)
        np.save(os.path.join(model_path, "generation_categories_"+ood_idxs_file), metrics)
        return {}
    
 
    dataset_orig = load_dataset('relbert/t_rex')
    ood_idxs = np.load(ood_idxs_file)
    ood_dataset = dataset_orig["train"].select(ood_idxs)


    prompts_ood = []

    # for prompttype in ["force_commit", "force_hedge"]:
    for prompttype in ["orig"]:
        template_sub_label_answer_split_prompttype = list(zip(ood_dataset["relation"], ood_dataset["head"], ood_dataset["tail"], [3 for _ in range(len(ood_dataset["relation"]))], [prompttype for _ in range(len(ood_dataset["relation"]))]))
        prompts_ood += list(map(prepare_prompt, template_sub_label_answer_split_prompttype))


    trainer = trlx.eval(
        eval_prompts=prompts_ood,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

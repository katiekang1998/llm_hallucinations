import json
import os
import sys
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_sft_config, default_ppo_config
import numpy as np

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

CORRECT_REWARD = 30 
ood_idxs_file = "trex_ood_idxs_128+.npy"

with open('trex_relations2questions2.json', 'rb') as fp:
    trex_relations2questions = json.load(fp)

def answer_type_individial(output , answer) -> List[float]:
    if output[-len("<|endoftext|>"):] == "<|endoftext|>":
        output = output[: -len("<|endoftext|>")]
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
    # model_path =  "ckpts/sft_lama_GPT2_commit/checkpoint_02000/hf_model"
    # model_path =  "ckpts/sft_lama_GPT2_commit_hedge_idk/checkpoint_10000/hf_model"
    model_path = "ckpts/ppo_lama_GPT2_3_commit30_hedge25.5_6.5_idk11_cr0.0005/checkpoint_80000/hf_model"
    # model_path =  "ckpts/sft2_lama_GPT2/checkpoint_10000/hf_model"

    # model_path = "ckpts/sft2_lama_GPT2_lr1e-4/checkpoint_80000/hf_model"
    if "sft" in model_path:
        config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    elif "ppo" in model_path:
        config = TRLConfig.update(default_ppo_config().to_dict(), hparams) 
    config.train.total_steps = 0
    config.train.eval_interval = 1
    config.model.model_path = model_path
    config.train.checkpoint_dir = "ckpts/delete"
    config.train.project_name = "delete"
    config.train.run_name = "delete"


    config.method.gen_kwargs=dict(max_new_tokens=40, do_sample=False)

    def metric_fn(samples: List[str], **kwargs):

        
        # answer_stylized = [" "+a+"." for a in kwargs["answer"]]

        # hedge_idxs = np.where(np.array(kwargs["prompttype"])=="force_hedge")[0]
        # train_idxs = np.where(np.array(kwargs["split"])[hedge_idxs]==0)[0]
        # train_accuracy = (np.array(kwargs["outputs"])[hedge_idxs][train_idxs] == np.array(answer_stylized)[hedge_idxs][train_idxs]).sum()/len(train_idxs)
        # test_idxs = np.where(np.array(kwargs["split"])[hedge_idxs]==2)[0]
        # test_accuracy = (np.array(kwargs["outputs"])[hedge_idxs][test_idxs] == np.array(answer_stylized)[hedge_idxs][test_idxs]).sum()/len(test_idxs)
        # ood_idxs = np.where(np.array(kwargs["split"])[hedge_idxs]==3)[0]
        # ood_accuracy = (np.array(kwargs["outputs"])[hedge_idxs][ood_idxs] == np.array(answer_stylized)[hedge_idxs][ood_idxs]).sum()/len(ood_idxs)
        # print("force hedge accuracies")
        # print(train_accuracy, test_accuracy, ood_accuracy)

        # commit_idxs = np.where(np.array(kwargs["prompttype"])=="force_commit")[0]

        # train_idxs = np.where(np.array(kwargs["split"])[commit_idxs]==0)[0]
        # train_accuracy = (np.array(kwargs["outputs"])[commit_idxs][train_idxs] == np.array(answer_stylized)[commit_idxs][train_idxs]).sum()/len(train_idxs)
        # test_idxs = np.where(np.array(kwargs["split"])[commit_idxs]==2)[0]
        # test_accuracy = (np.array(kwargs["outputs"])[commit_idxs][test_idxs] == np.array(answer_stylized)[commit_idxs][test_idxs]).sum()/len(test_idxs)
        # ood_idxs = np.where(np.array(kwargs["split"])[commit_idxs]==3)[0]
        # ood_accuracy = (np.array(kwargs["outputs"])[commit_idxs][ood_idxs] == np.array(answer_stylized)[commit_idxs][ood_idxs]).sum()/len(ood_idxs)
        # print("force commit accuracies")
        # print(train_accuracy, test_accuracy, ood_accuracy)

        import IPython; IPython.embed()


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
        # eval_prompts=prompts_train,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["<|endoftext|>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
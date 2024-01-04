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

CORRECT_REWARD = 50

#load json
with open('trex_relations2questions2_cleaned.json', 'rb') as fp:
    template2question = json.load(fp)

#load json
with open('trex_relations2questions2_cleaned.json', 'rb') as fp:
    template2question = json.load(fp)


def answer_type_individial(output ) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    if (output == " Yes."):
        answer_type = 0
    elif (output == " No."):
        answer_type = 1
    else:
        answer_type = 2
    return answer_type

def convert_template_to_prompt(template_sub_label_answer):
    template, sub_label, answer = template_sub_label_answer
    assert(template in template2question.keys())
    question = template2question[template][1]
    question = question.replace("[X]", sub_label)
    prompt = question + " The answer is " + answer + ". Is the answer to the question correct?"
    return prompt

def prepare_prompt(line):
    prompt = {}
    prompt["prompt"] = line + " Is the answer to the question correct?"
    prompt["answer"] = " None"
    return prompt

def prepare_prompt_yes(template_sub_label_answer_split):
    template, sub_label, answer, split = template_sub_label_answer_split
    prompt_str = convert_template_to_prompt((template, sub_label, answer))
    prompt = {}
    prompt["prompt"] = prompt_str
    prompt["answer"] = " Yes."
    prompt["split"] = split
    return prompt

def prepare_prompt_no(template_sub_label_answer_split):
    template, sub_label, answer, split = template_sub_label_answer_split
    prompt_str = convert_template_to_prompt((template, sub_label, answer))
    prompt = {}
    prompt["prompt"] = prompt_str
    prompt["answer"] = " No."
    prompt["split"] = split
    return prompt


def main(hparams={}):
    model_path = "ckpts/rm_ctrex_llama7B_certain_only/checkpoint_10000/hf_model/"

    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.model.model_path = model_path

    config.train.batch_size = 32

    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    if "special_tokens" in model_path:
        config.tokenizer.additional_special_tokens = ['<True>', '<False>']

    
    config.method.gen_kwargs=dict(max_new_tokens=40, do_sample=False)


    # config.model.peft_config = LoraConfig(
    #     r=16,
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_alpha=16,
    #     lora_dropout=0, 
    # )

    def metric_fn(samples: List[str], **kwargs):
        # answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])))
        # np.save(os.path.join(model_path, "ppo_rm_ctrex_llama7B_commit30_idk10_answer_types.npy"), answer_types)
        return {}    


    # generated_responses = np.load("ckpts/sft_ctrex_llama7B_2_commit_lr1e-5_2/checkpoint_30000/hf_model/output_strings_oodsmall.npy")

    # lines_all = []
    # for response in generated_responses:
    #     if '<unk> ' in response:
    #         line = response.split('<unk> ')[1]
    #     line = line.split(' Label: ')[0].strip()
    #     lines_all.append(line)
    
    # prompts = list(map(prepare_prompt, lines_all))




    dataset_orig = load_dataset('relbert/t_rex')
    ood_idxs = np.load("custom_trex/ood_points_small.npy")
    ood_dataset = dataset_orig["train"].select(ood_idxs)
    ood_incorrect_tails = np.load("custom_trex/incorrect_tails/ood_small_incorrect_tails.npy")

    template_sub_label_answer_ood = list(zip(ood_dataset["relation"], ood_dataset["head"], ood_dataset["tail"], [3 for _ in range(len(ood_dataset["relation"]))]))
    prompts_ood_yes = list(map(prepare_prompt_yes, template_sub_label_answer_ood))

    template_sub_label_answer_ood = list(zip(ood_dataset["relation"], ood_dataset["head"], ood_incorrect_tails, [3 for _ in range(len(ood_dataset["relation"]))]))
    prompts_ood_no = list(map(prepare_prompt_no, template_sub_label_answer_ood))

    prompts_ood = prompts_ood_yes+prompts_ood_no



    trainer = trlx.eval(
        eval_prompts=prompts_ood,
        # eval_prompts=prompts_train,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

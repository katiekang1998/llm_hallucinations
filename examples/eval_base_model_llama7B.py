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

import tqdm
import torch
import re
import string


from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)


def prepare_prompt(name):
    prompt = {}
    prompt["prompt"] = "Write a biography for "+name+"."
    prompt["name"] = name
    return prompt


# def prepare_prompt(question, choices, answer, split):
#     letters = ["A", "B", "C", "D"]

#     prompt = question + " "
#     for i, choice in enumerate(choices):
#         prompt += letters[i] + ") " + choice + " "

#     prompt += "\nAnswer: "

#     prompt_dict = {}
#     # prompt_dict["prompt"] = prompt
#     prompt_dict["prompt"] = "Q: "+question
#     prompt_dict["answer"] = letters[answer]
#     prompt_dict["split"] = split
#     return prompt_dict

def main(hparams={}):
    model_path = "NousResearch/Llama-2-7b-hf"
    # model_path = "ckpts/sft_input_mmlu_llama7B/checkpoint_05000/hf_model/"


    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.model.model_path = model_path

    config.train.batch_size = 1


    # config.train.epochs = 100
    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"


    def metric_fn(samples: List[str], **kwargs):
        return {}


    def eval_fn(eval_dataloader, model, tokenizer, device, config, accelerator):

        print("EVALUATING")

        perplexity_all = []
        for i_prompt, prompts in enumerate(tqdm.tqdm(eval_dataloader)):
            outputs = model(input_ids= prompts["input_ids"], attention_mask = prompts["attention_mask"], labels = prompts["input_ids"])

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = prompts["input_ids"][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduce=False)

            loss = loss_fct(shift_logits.swapaxes(-1, -2), shift_labels)
            perplexity = np.e**(loss.sum(axis=1)/(loss!=0).sum(axis=1))

            perplexity_all.append(perplexity.tolist())
        perplexity_all = np.concatenate(perplexity_all)
        np.save(os.path.join("base_model_perplexities", "bios_test_points_medium.npy"), perplexity_all)
        
        # perplexity_all = accelerator.gather(perplexity_all)
        # if accelerator.is_main_process:
        #     import IPython; IPython.embed(); exit(1)
        #     perplexity_all = np.concatenate(perplexity_all)
        #     np.save(os.path.join("base_model_perplexities", "mmlu_test.npy"), perplexity_all)
        


    names = np.load("biographies/names.npy")

    test_idxs = np.load("biographies/test_points_medium.npy")

    prompts_test = list(map(prepare_prompt, names[test_idxs]))


    # topics = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

    # test_questions = []
    # test_choices = []
    # test_answers = []
    # for topic in topics:
    #     dataset = load_dataset("tasksource/mmlu", topic)
    #     test_questions.append(dataset["validation"]["question"])
    #     test_choices.append(dataset["validation"]["choices"])
    #     test_answers.append(dataset["validation"]["answer"])
    # test_questions = np.concatenate(test_questions)
    # test_choices = np.concatenate(test_choices)
    # test_answers = np.concatenate(test_answers)
    # prompts_test = list(map(prepare_prompt, test_questions,test_choices,test_answers, [2 for _ in range(len(test_questions))]))


    trainer = trlx.eval(
        eval_prompts=prompts_test,
        # metric_fn=metric_fn,
        eval_fn = eval_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

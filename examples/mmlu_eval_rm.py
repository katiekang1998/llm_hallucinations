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
import torch


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
    if output in ["True", "False"]:
        if output == answer:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


def prepare_prompt_correct(question, choices, answer, split):
    letters = ["A", "B", "C", "D"]

    prompt = question + " "
    for i, choice in enumerate(choices):
        prompt += letters[i] + ") " + choice + " "

    prompt += ", Answer: "+letters[answer]+"\n"

    prompt_dict = {}
    prompt_dict["prompt"] = prompt
    prompt_dict["answer"] = "True"
    prompt_dict["split"] = split
    return prompt_dict

def prepare_prompt_incorrect(question, choices, answer, split):
    letters = ["A", "B", "C", "D"]

    prompt = question + " "
    for i, choice in enumerate(choices):
        prompt += letters[i] + ") " + choice + " "
        
    incorrect_letters = np.delete(letters, answer)
    incorrect_letter = np.random.choice(incorrect_letters)

    prompt += ", Answer: "+incorrect_letter+"\n"

    prompt_dict = {}
    prompt_dict["prompt"] = prompt
    prompt_dict["answer"] = "False"
    prompt_dict["split"] = split
    return prompt_dict

def main(hparams={}):
    # EVAL_TYPE = "correct_only"
    EVAL_TYPE = "correct_only"


    # model_path = "ckpts/rm_mmlu_unfamiliar_correct/checkpoint_10000/hf_model/"
    model_path = "ckpts/rm_mmlu_unfamiliar_incorrect/checkpoint_10000/hf_model/"

    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.model.model_path = model_path

    config.train.batch_size = 4

    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    
    config.method.gen_kwargs=dict(max_new_tokens=40, do_sample=False) 


    def eval_fn(eval_dataloader, model, tokenizer, device, config, accelerator):
        True_False_tokens = np.array(tokenizer(["True", "False"], add_special_tokens=False)["input_ids"]).squeeze()
        True_False_logits_all = []
        with torch.no_grad():
            for i_prompt, prompts in enumerate(eval_dataloader):
                outputs = model(input_ids= prompts["input_ids"], attention_mask = prompts["input_ids"]!=tokenizer.pad_token_id)
                logits  = outputs.logits[:, -1, True_False_tokens]
                # logits = logits.softmax(dim=-1)
                True_False_logits_all.append(accelerator.gather_for_metrics(accelerator.pad_across_processes(logits)))

        if accelerator.is_main_process:
            True_False_logits_all = torch.cat(True_False_logits_all, axis=0).cpu().numpy()
            np.save(os.path.join(config.model.model_path, f"test_{EVAL_TYPE}_logits.npy"), True_False_logits_all)


    topics = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    
    train_questions = []
    train_choices = []
    train_answers = []

    test_questions = []
    test_choices = []
    test_answers = []
    for topic in topics:
        dataset = load_dataset("tasksource/mmlu", topic)
        train_questions.append(dataset["test"]["question"])
        train_choices.append(dataset["test"]["choices"])
        train_answers.append(dataset["test"]["answer"])
        test_questions.append(dataset["validation"]["question"])
        test_choices.append(dataset["validation"]["choices"])
        test_answers.append(dataset["validation"]["answer"])
    train_questions = np.concatenate(train_questions)
    train_choices = np.concatenate(train_choices)
    train_answers = np.concatenate(train_answers)
    test_questions = np.concatenate(test_questions)
    test_choices = np.concatenate(test_choices)
    test_answers = np.concatenate(test_answers)
    
    if EVAL_TYPE == "correct_only":
        prompts_test = list(map(prepare_prompt_correct, test_questions,test_choices,test_answers, [0 for _ in range(len(test_questions))]))
    elif EVAL_TYPE == "incorrect_only":
        prompts_test = list(map(prepare_prompt_incorrect, test_questions,test_choices,test_answers, [0 for _ in range(len(test_questions))]))
    else:
        raise Exception("invalid EVAL_TYPE")
    
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

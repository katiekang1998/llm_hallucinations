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

import tqdm

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)






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

        # answer_log_probs_mean_all = []
        # for i_prompt, prompts in enumerate(eval_dataloader):
        #     # labels = torch.Tensor(tokenizer(prompts["answer"], add_special_tokens=False)["input_ids"]).int().to(device)
        #     # samples = torch.cat([prompts["input_ids"], labels], dim=1)

        #     samples = model.generate(input_ids=prompts["input_ids"], attention_mask = prompts["attention_mask"], **config.method.gen_kwargs)
        #     labels = samples.clone()
        #     labels[:,:prompts["input_ids"].shape[1]] = tokenizer.pad_token_id
        #     outputs = model(input_ids= samples, attention_mask = samples!=tokenizer.pad_token_id, labels = labels)
        #     shift_logits = outputs.logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduce=False)

        #     loss = loss_fct(shift_logits.swapaxes(-1, -2), shift_labels)
        #     log_likelihood = -loss.sum(axis=1)

        #     answer_log_probs_mean_all.append(log_likelihood.tolist())

        # answer_log_probs_mean_all = np.concatenate(answer_log_probs_mean_all)
        # np.save(os.path.join(config.model.model_path, "eval_log_probs2.npy"), answer_log_probs_mean_all)
        

        A_to_E_tokens = [319, 350, 315, 360]
        A_to_E_logits_all = []

        answers = []

        for i_prompt, prompts in enumerate(tqdm.tqdm(eval_dataloader)):
            outputs = model(input_ids= prompts["input_ids"], attention_mask = prompts["input_ids"]!=tokenizer.pad_token_id)
            logits  = outputs.logits[:, -1, A_to_E_tokens]
            logits = logits.softmax(dim=-1)
            A_to_E_logits_all.append(logits.tolist())
            answers.append(prompts["answer"])

        answers = np.concatenate(answers)
        np.save(os.path.join("base_model_MMLU", "train_answers.npy"), answers)

        A_to_E_logits_all = np.concatenate(A_to_E_logits_all, axis=0)
        np.save(os.path.join("base_model_MMLU", "train_A_to_D_probs.npy"), A_to_E_logits_all)


    topics = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

    split = "test"

    test_questions = []
    test_choices = []
    test_answers = []
    test_subjects = []
    dev_dict = {}
    for topic in topics:
        dataset = load_dataset("tasksource/mmlu", topic)
        test_questions.append(dataset[split]["question"])
        test_choices.append(dataset[split]["choices"])
        test_answers.append(dataset[split]["answer"])
        test_subjects.append([topic for _ in range(len(dataset[split]["question"]))])
        dev_dict[topic] = dataset["dev"]
    test_questions = np.concatenate(test_questions)
    test_choices = np.concatenate(test_choices)
    test_answers = np.concatenate(test_answers)
    test_subjects = np.concatenate(test_subjects)


    def process_item(questions, choices, answers,):
        keys = ['A', 'B', 'C', 'D']
        question = questions
        choices = ''.join([f"{key}. {choice}\n" for choice, key in zip(choices, keys)])
        prompt = f"{question}\n{choices}Answer:"
        target = ' ' + keys[answers]
        return prompt, target

    def create_prompt_for_item(questions, choices, answers, subject, shots):
        subject_name = " ".join(subject.split('_'))
        description = f"The following are multiple choice questions (with answers) about {subject_name}."
        prompt = f"{description}\n\n"
        for shot in shots:
            shot_question, shot_choices, shot_answers = shot["question"], shot["choices"], shot["answer"]
            shot_prompt, shot_target = process_item(shot_question, shot_choices, shot_answers,)
            prompt += f"{shot_prompt}{shot_target}\n\n"
        item_prompt, _ = process_item(questions, choices, answers,)
        prompt += f"{item_prompt}"
        return prompt

    def get_fewshot_for_example(questions, choices, answers, subject, n_shot):
        fewshot_items = dev_dict[subject]
        fewshot_items = list(fewshot_items)[:n_shot]
        return create_prompt_for_item(questions, choices, answers, subject, fewshot_items)


    def prepare_prompt(questions, choices, answers, subject,):
        prompt = get_fewshot_for_example(questions, choices, answers, subject, n_shot=5)
        prompt_dict = {}
        prompt_dict["prompt"] =prompt
        prompt_dict["answer"] = answers
        return prompt_dict


    prompts_test = list(map(prepare_prompt, test_questions,test_choices,test_answers, test_subjects))

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

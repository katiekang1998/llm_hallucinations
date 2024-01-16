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
    if output in ["A", "B", "C", "D"]:
        if output == answer:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


def prepare_prompt(question, choices, answer, split):
    letters = ["A", "B", "C", "D"]

    prompt = question + " "
    for i, choice in enumerate(choices):
        prompt += letters[i] + ") " + choice + " "

    prompt += "\nAnswer: "

    prompt_dict = {}
    prompt_dict["prompt"] = prompt
    prompt_dict["answer"] = letters[answer]
    prompt_dict["split"] = split
    return prompt_dict

def prepare_prompt_random(question, correct_answer, incorrect_answers):

    if np.random.random() < 0.25:
        answers = [correct_answer, incorrect_answers[0], incorrect_answers[1], incorrect_answers[2]]
        answer_idx = 0
    elif np.random.random() < 0.5:
        answers = [incorrect_answers[0], correct_answer, incorrect_answers[1], incorrect_answers[2]]
        answer_idx = 1
    elif np.random.random() < 0.75:
        answers = [incorrect_answers[0], incorrect_answers[1], correct_answer, incorrect_answers[2]]
        answer_idx = 2
    else:
        answers = [incorrect_answers[0], incorrect_answers[1], incorrect_answers[2], correct_answer]
        answer_idx = 3
    
    choices = ["A", "B", "C", "D"]

    prompt = question + " "
    for i, answer in enumerate(answers):
        prompt += choices[i] + ") " + answer + " "

    prompt += ", Answer: "

    response = choices[answer_idx]

    prompt_dict = {}
    prompt_dict["prompt"] = prompt
    prompt_dict["answer"] = response

    return prompt_dict

def prepare_prompt_letter(question, correct_answer, incorrect_answers, letter):

    if letter=="A":
        answers = [correct_answer, incorrect_answers[0], incorrect_answers[1], incorrect_answers[2]]
        answer_idx = 0
    elif letter=="B":
        answers = [incorrect_answers[0], correct_answer, incorrect_answers[1], incorrect_answers[2]]
        answer_idx = 1
    elif letter=="C":
        answers = [incorrect_answers[0], incorrect_answers[1], correct_answer, incorrect_answers[2]]
        answer_idx = 2
    elif letter=="D":
        answers = [incorrect_answers[0], incorrect_answers[1], incorrect_answers[2], correct_answer]
        answer_idx = 3
    else:
        raise Exception("Letter not recognized")
    
    choices = ["A", "B", "C", "D"]

    prompt = question + " "
    for i, answer in enumerate(answers):
        prompt += choices[i] + ") " + answer + " "

    prompt += ", Answer: "

    response = choices[answer_idx]

    prompt_dict = {}
    prompt_dict["prompt"] = prompt
    prompt_dict["answer"] = response

    return prompt_dict

def main(hparams={}):
    # model_path = "ckpts/sft2_mmlu_llama7B_threshold0pt5_certainABCD/checkpoint_01000/hf_model/"
    model_path = "ckpts/sft_mmlu_llama7B/checkpoint_02000/hf_model/"


    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.model.model_path = model_path

    config.train.batch_size = 4

    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    
    config.method.gen_kwargs=dict(max_new_tokens=40, do_sample=False) 

    def metric_fn(samples: List[str], **kwargs):
        # np.save(os.path.join(config.model.model_path, "eval_answers.npy"), np.array(kwargs["answer"]))
        
        output_dict = {}
        
        # answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"]), np.array(kwargs["answer"])))
        # correct_pred = ([1 if x == 0 else 0 for x in answer_types ])
        # incorrect_pred = ([1 if x == 1 else 0 for x in answer_types ])
        # bad_pred = ([1 if x == 4 else 0 for x in answer_types ])
    
        # total = len(answer_types)
        
        # output_dict["eval/correct_pred"] = np.sum(correct_pred)/total
        # output_dict["eval/incorrect_pred"] = np.sum(incorrect_pred)/total
        # output_dict["eval/bad_pred"] = np.sum(bad_pred)/total

        # print(output_dict)

        # np.save(os.path.join(config.model.model_path, "train_correct_preds.npy"), correct_pred)
        return output_dict
    

    def eval_fn(eval_dataloader, model, tokenizer, device, config):

        print("EVALUATING")


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


        A_to_D_tokens = [319, 350, 315, 360]
        A_to_D_logits_all = []

        answers = []

        for i_prompt, prompts in enumerate(eval_dataloader):
            outputs = model(input_ids= prompts["input_ids"], attention_mask = prompts["input_ids"]!=tokenizer.pad_token_id)
            logits  = outputs.logits[:, -1, A_to_D_tokens]
            logits = logits.softmax(dim=-1)
            A_to_D_logits_all.append(logits.tolist())
            answers.append(prompts["answer"])

        answers = np.concatenate(answers)
        np.save(os.path.join(config.model.model_path, "eval_answers.npy"), answers)

        A_to_D_logits_all = np.concatenate(A_to_D_logits_all, axis=0)
        np.save(os.path.join(config.model.model_path, "eval_A_to_D_probs_truthD.npy"), A_to_D_logits_all)

    

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


    train_correct_choice = []
    train_incorrect_choices = []
    for i in range(len(train_questions)):
        train_correct_choice.append(train_choices[i][train_answers[i]])
        train_incorrect_choices_i = np.delete(train_choices[i], train_answers[i])
        np.random.shuffle(train_incorrect_choices_i)
        train_incorrect_choices.append(train_incorrect_choices_i)
    
    test_correct_choice = []
    test_incorrect_choices = []
    for i in range(len(test_questions)):
        test_correct_choice.append(test_choices[i][test_answers[i]])
        test_incorrect_choices_i = np.delete(test_choices[i], test_answers[i])
        np.random.shuffle(test_incorrect_choices_i)
        test_incorrect_choices.append(test_incorrect_choices_i)

    
    # prompts_test = list(map(prepare_prompt, test_questions,test_choices,test_answers, [2 for _ in range(len(test_questions))]))
    prompts_test = list(map(prepare_prompt_letter, test_questions, test_correct_choice, test_incorrect_choices, ["D" for _ in range(len(test_questions))]))

    # prompts_test = list(map(prepare_prompt, train_questions,train_choices,train_answers, [2 for _ in range(len(train_questions))]))



    trainer = trlx.eval(
        eval_prompts=prompts_test,
        # eval_prompts=prompts_train,
        metric_fn=metric_fn,
        eval_fn = eval_fn,
        config=config,
        stop_sequences = ["</s>"]
    )

if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

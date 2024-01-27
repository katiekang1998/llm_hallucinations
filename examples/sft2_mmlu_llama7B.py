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


# def prepare_sample(question, choices, answer):
#     letters = ["A", "B", "C", "D"]

#     prompt = question + " "
#     for i, choice in enumerate(choices):
#         prompt += letters[i] + ") " + choice + " "

#     prompt += "\nAnswer: "
#     response = letters[answer]

#     return (prompt, response)

def prepare_sample_ABCD(question, correct_answer, incorrect_answers):

    rand_num = np.random.random()

    if rand_num < 0.25:
        answers = [correct_answer, incorrect_answers[0], incorrect_answers[1], incorrect_answers[2]]
        answer_idx = 0
    elif rand_num < 0.5:
        answers = [incorrect_answers[0], correct_answer, incorrect_answers[1], incorrect_answers[2]]
        answer_idx = 1
    elif rand_num < 0.75:
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

    return (prompt, response)

def prepare_sample_AD(question, correct_answer, incorrect_answers):

    if np.random.random() < 0.5:
        answers = [correct_answer, incorrect_answers[0], incorrect_answers[1], incorrect_answers[2]]
        answer_idx = 0
    else:
        answers = [incorrect_answers[0], incorrect_answers[1], incorrect_answers[2], correct_answer]
        answer_idx = 3
    
    choices = ["A", "B", "C", "D"]

    prompt = question + " "
    for i, answer in enumerate(answers):
        prompt += choices[i] + ") " + answer + " "

    prompt += ", Answer: "

    response = choices[answer_idx]

    return (prompt, response)

def prepare_sample_BC(question, correct_answer, incorrect_answers):

    if np.random.random() < 0.5:
        answers = [ incorrect_answers[0], correct_answer, incorrect_answers[1], incorrect_answers[2]]
        answer_idx = 1
    else:
        answers = [incorrect_answers[0], incorrect_answers[1], correct_answer, incorrect_answers[2]]
        answer_idx = 2
    
    choices = ["A", "B", "C", "D"]

    prompt = question + " "
    for i, answer in enumerate(answers):
        prompt += choices[i] + ") " + answer + " "

    prompt += ", Answer: "

    response = choices[answer_idx]

    return (prompt, response)


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

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 30000
    config.train.eval_interval = 500
    config.train.checkpoint_interval = 500
    config.train.checkpoint_dir = "ckpts/sft2_mmlu_llama7B_threshold0pt5_certainABCD_2"
    # config.train.epochs = 100
    config.train.batch_size = 2
    config.train.project_name = "sft_mmlu_llama7B"
    config.train.run_name = "sft2_threshold0pt5_certainABCD_2"

    config.model.model_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
    config.scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e4, eta_min=1e-10)  # train.total_steps
        )

    config.model.peft_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0,
    )

    def metric_fn(samples: List[str], **kwargs):
        split_names = ["eval_certain", "eval_uncertain"]
        output_dict = {}

        for split_idx in range(len(split_names)):
            idxs = np.where(np.array(kwargs["split"])==split_idx)[0]
            
            answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[idxs], np.array(kwargs["answer"])[idxs]))
            correct_pred = ([1 if x == 0 else 0 for x in answer_types ])
            incorrect_pred = ([1 if x == 1 else 0 for x in answer_types ])
            bad_pred = ([1 if x == 4 else 0 for x in answer_types ])
        
            total = len(answer_types)

            filtered_outputs = np.array(kwargs["outputs"])[idxs]
            for i, output in enumerate(filtered_outputs):
                if output[-len(" </s>"):] == " </s>":
                    filtered_outputs[i] = output[: -len(" </s>")]
                if output[-len("</s>"):] == "</s>":
                    filtered_outputs[i] = output[: -len("</s>")]
            
            output_dict[split_names[split_idx]+"/correct_pred"] = np.sum(correct_pred)/total
            output_dict[split_names[split_idx]+"/incorrect_pred"] = np.sum(incorrect_pred)/total
            output_dict[split_names[split_idx]+"/bad_pred"] = np.sum(bad_pred)/total
            output_dict[split_names[split_idx]+"/A_frac"] = np.sum(filtered_outputs == "A")/total
            output_dict[split_names[split_idx]+"/B_frac"] = np.sum(filtered_outputs == "B")/total
            output_dict[split_names[split_idx]+"/C_frac"] = np.sum(filtered_outputs == "C")/total
            output_dict[split_names[split_idx]+"/D_frac"] = np.sum(filtered_outputs == "D")/total
        return output_dict
    


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

    # eval_train_idxs = np.random.choice(len(train_questions), 1000, replace=False)

    train_correct_choice = []
    train_incorrect_choices = []
    for i in range(len(train_questions)):
        train_correct_choice.append(train_choices[i][train_answers[i]])
        train_incorrect_choices_i = np.delete(train_choices[i], train_answers[i])
        np.random.shuffle(train_incorrect_choices_i)
        train_incorrect_choices.append(train_incorrect_choices_i)


    certain_idxs = np.where(np.load("ckpts/sft_mmlu_llama7B/checkpoint_01000/hf_model/train_A_to_D_probs.npy").max(axis=1) > 0.5)[0]
    uncertain_idxs = np.where(np.load("ckpts/sft_mmlu_llama7B/checkpoint_01000/hf_model/train_A_to_D_probs.npy").max(axis=1) <= 0.5)[0]
    num_idxs = len(certain_idxs)
    uncertain_idxs = np.random.choice(uncertain_idxs, size=num_idxs, replace=False)
    train_questions = np.array(train_questions)
    train_correct_choice = np.array(train_correct_choice)
    train_incorrect_choices = np.array(train_incorrect_choices)

    train_samples_certain = list(map(prepare_sample_ABCD, train_questions[certain_idxs], train_correct_choice[certain_idxs], train_incorrect_choices[certain_idxs]))
    train_samples_uncertain = list(map(prepare_sample_BC, train_questions[uncertain_idxs], train_correct_choice[uncertain_idxs], train_incorrect_choices[uncertain_idxs]))
    train_samples = train_samples_certain + train_samples_uncertain
    np.random.shuffle(train_samples)

    # prompts_eval_train = list(map(prepare_prompt, train_questions[eval_train_idxs], train_choices[eval_train_idxs], train_answers[eval_train_idxs], [1 for _ in range(len(eval_train_idxs))]))
    
    certain_idxs = np.where(np.load("ckpts/sft_mmlu_llama7B/checkpoint_01000/hf_model/eval_A_to_D_probs.npy").max(axis=1) > 0.5)[0]
    uncertain_idxs = np.where(np.load("ckpts/sft_mmlu_llama7B/checkpoint_01000/hf_model/eval_A_to_D_probs.npy").max(axis=1) <= 0.5)[0]
    test_questions = np.array(test_questions)
    test_choices = np.array(test_choices)
    test_answers = np.array(test_answers)

    prompts_test_certain = list(map(prepare_prompt, test_questions[certain_idxs],test_choices[certain_idxs],test_answers[certain_idxs], [0 for _ in range(len(certain_idxs))]))
    prompts_test_uncertain = list(map(prepare_prompt, test_questions[uncertain_idxs],test_choices[uncertain_idxs],test_answers[uncertain_idxs], [1 for _ in range(len(uncertain_idxs))]))
    prompts_test = prompts_test_certain + prompts_test_uncertain
    # prompts_eval = prompts_eval_train+prompts_test


    trainer = trlx.train(
        samples=train_samples,
        eval_prompts=prompts_test,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

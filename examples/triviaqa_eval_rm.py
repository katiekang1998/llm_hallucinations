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
import string
import re
from tqdm import tqdm


from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


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


def prepare_prompt_correct(question, answer, split):
    prompt_dict = {}
    prompt_dict["prompt"] = "Q: "+ question + "\nA: " + normalize_answer(answer) + "\n"
    prompt_dict["answer"] = "True"
    prompt_dict["split"] = split
    return prompt_dict

def prepare_prompt_incorrect(question, answer, split):
    prompt_dict = {}
    prompt_dict["prompt"] = "Q: "+ question + "\nA: " + normalize_answer(answer) + "\n"
    prompt_dict["answer"] = "False"
    prompt_dict["split"] = split
    return prompt_dict

def main(hparams={}):
    # EVAL_TYPE = "correct_only"
    EVAL_TYPE = "incorrect_only"


    model_path = "ckpts/rm_triviaqa_unfamiliar_correct_2/checkpoint_08000/hf_model/"

    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.model.model_path = model_path

    config.train.batch_size = 32//3

    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    
    config.method.gen_kwargs=dict(max_new_tokens=40, do_sample=False) 


    def eval_fn(eval_dataloader, model, tokenizer, device, config, accelerator):
        True_False_tokens = np.array(tokenizer(["True", "False"], add_special_tokens=False)["input_ids"]).squeeze()
        True_False_logits_all = []
        with torch.no_grad():
            for i_prompt, prompts in enumerate(tqdm(eval_dataloader)):
                outputs = model(input_ids= prompts["input_ids"], attention_mask = prompts["input_ids"]!=tokenizer.pad_token_id)
                logits  = outputs.logits[:, -1, True_False_tokens]
                # logits = logits.softmax(dim=-1)
                True_False_logits_all.append(accelerator.gather_for_metrics(accelerator.pad_across_processes(logits)))

        if accelerator.is_main_process:
            True_False_logits_all = torch.cat(True_False_logits_all, axis=0).cpu().numpy()
            np.save(os.path.join(config.model.model_path, f"test_{EVAL_TYPE}_logits.npy"), True_False_logits_all)

    dataset_orig = load_dataset("trivia_qa", "rc.nocontext")

    dataset = dataset_orig["train"]
    test_dataset = dataset_orig["validation"]
    
    
    train_questions = dataset["question"]
    train_correct_answer = [answer["value"] for answer in dataset["answer"]]
    train_incorrect_answer = np.load("ckpts/triviaqa_incorrect_answers/train_false_answers2.npy")
    
    train_questions = np.array(train_questions)
    train_correct_answer = np.array(train_correct_answer)
    train_incorrect_answer = np.array(train_incorrect_answer)
    
    test_questions = test_dataset["question"]
    test_correct_answer = [answer["value"] for answer in test_dataset["answer"]]
    test_incorrect_answer = np.load("ckpts/triviaqa_incorrect_answers/test_false_answers2.npy")
    
    test_questions = np.array(test_questions)
    test_correct_answer = np.array(test_correct_answer)
    test_incorrect_answer = np.array(test_incorrect_answer)
    
    
    if EVAL_TYPE == "correct_only":
        prompts_test = list(map(prepare_prompt_correct, test_questions,test_correct_answer, [0 for _ in range(len(test_questions))]))
    elif EVAL_TYPE == "incorrect_only":
        prompts_test = list(map(prepare_prompt_incorrect, test_questions,test_incorrect_answer, [0 for _ in range(len(test_questions))]))
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

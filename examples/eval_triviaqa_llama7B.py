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

def answer_type_individial(output , answer, aliases) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]


    aliases_normalized = []
    for alias in aliases:
        aliases_normalized.append(normalize_answer(alias))

    if output[:len(" The answer is ")] == " The answer is ":
        predicted_answer = output[len(" The answer is "):-1]
        if normalize_answer(predicted_answer) == normalize_answer(answer) or normalize_answer(predicted_answer) in aliases_normalized:
            answer_type = 0
        else:
            answer_type = 1
    elif output == " I don't know.":
        answer_type = 2
    else:
        answer_type = 3
    return answer_type


def prepare_sample(point):
    return (point["question"], " The answer is "+ point["answer"]["value"]+".")

def prepare_prompt(point):
    prompt = {}
    prompt["prompt"] = point["question"]
    prompt["answer"] = point["answer"]["value"]
    prompt["aliases"] = point["answer"]["aliases"]
    return prompt


def main(hparams={}):
    model_path = "ckpts/sft2_triviaqa_llama7B_t0.85/checkpoint_20000/hf_model/"
    # model_path = "ckpts/sft_triviaqa_llama7B/checkpoint_20000/hf_model/"

    if "sft" in model_path:
        config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    elif "ppo" in model_path:
        config = TRLConfig.update(default_ppo_config().to_dict(), hparams) 
        config.method.chunk_size = 32//3
    config.model.model_path = model_path

    config.train.batch_size = 32//3


    # config.train.epochs = 100
    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1.0e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
    config.scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e4, eta_min=1.0e-10)  # train.total_steps
        )
    
    config.method.gen_kwargs=dict(max_new_tokens=50, do_sample=False)


    # config.model.peft_config = LoraConfig(
    #     r=16,
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_alpha=16,
    #     lora_dropout=0,
    # )


    def metric_fn(samples: List[str], **kwargs):
        np.save(os.path.join(model_path, "test_output_strings.npy"), kwargs["outputs"])

        output_dict = {}
        answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"]), np.array(kwargs["answer"]), (kwargs["aliases"])))
        
        commit_correct = ([1 if x == 0 else 0 for x in answer_types ])
        commit_wrong = ([1 if x == 1 else 0 for x in answer_types ])
        dont_know = ([1 if x == 2 else 0 for x in answer_types ])
        wrong = ([1 if x == 3 else 0  for x in answer_types])

        total = len(answer_types)
        
        output_dict["test/commit_correct"] = np.sum(commit_correct)/total
        output_dict["test/commit_wrong"] = np.sum(commit_wrong)/total
        output_dict["test/dont_know"] = np.sum(dont_know)/total
        output_dict["test/wrong"] = np.sum(wrong)/total

        np.save(os.path.join(model_path, "test_answers_correct.npy"), commit_correct)
        np.save(os.path.join(model_path, "test_answers_dont_know.npy"), dont_know)

        return output_dict


    def eval_fn(eval_dataloader, model, tokenizer, device, config):

        print("EVALUATING")

        answer_log_probs_mean_all = []
        for i_prompt, prompts in enumerate(tqdm.tqdm(eval_dataloader)):
            # labels = torch.Tensor(tokenizer(prompts["answer"], add_special_tokens=False)["input_ids"]).int().to(device)
            # samples = torch.cat([prompts["input_ids"], labels], dim=1)

            samples = model.generate(input_ids=prompts["input_ids"], attention_mask = prompts["attention_mask"], **config.method.gen_kwargs)
            labels = samples.clone()
            labels[:,:prompts["input_ids"].shape[1]] = tokenizer.pad_token_id
            outputs = model(input_ids= samples, attention_mask = samples!=tokenizer.pad_token_id, labels = labels)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduce=False)

            loss = loss_fct(shift_logits.swapaxes(-1, -2), shift_labels)
            log_likelihood = -loss.sum(axis=1)/(loss!=0).sum(axis=1)

            answer_log_probs_mean_all.append(log_likelihood.tolist())
        

        answer_log_probs_mean_all = np.concatenate(answer_log_probs_mean_all)
        np.save(os.path.join(config.model.model_path, "test_answer_log_probs_mean_all.npy"), answer_log_probs_mean_all)
        

    dataset_orig = load_dataset("trivia_qa", "unfiltered.nocontext")

    dataset = dataset_orig["train"]
    test_dataset = dataset_orig["validation"]

    prompts_test = list(map(prepare_prompt, test_dataset))

    prompts_train = list(map(prepare_prompt, dataset))

    trainer = trlx.eval(
        eval_prompts=prompts_test,
        metric_fn=metric_fn,
        # eval_fn = eval_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

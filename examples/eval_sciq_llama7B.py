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

def prepare_prompt(element):
    answers = [element["correct_answer"], element["distractor1"], element["distractor2"], element["distractor3"], ]
    
    shuffle_idxs = np.random.choice(4, 4, replace=False)
    answer_idx = np.where(shuffle_idxs==0)[0][0]

    answers = [answers[i] for i in shuffle_idxs]
    choices = ["A", "B", "C", "D"]

    prompt = element["question"] + " "
    for i, answer in enumerate(answers):
        prompt += choices[i] + ") " + answer + " "

    prompt += ", Answer: "

    response = choices[answer_idx]


    prompt_dict = {}
    prompt_dict["prompt"] = prompt
    prompt_dict["answer"] = response
    return prompt_dict

def main(hparams={}):
    model_path = "ckpts/sft_sciq_llama7B/checkpoint_01500/hf_model/"

    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.model.model_path = model_path

    config.train.batch_size = 64

    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    
    config.method.gen_kwargs=dict(max_new_tokens=40, do_sample=False)

    def metric_fn(samples: List[str], **kwargs):
        output_dict = {}
        
        answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"]), np.array(kwargs["answer"])))
        correct_pred = ([1 if x == 0 else 0 for x in answer_types ])
        incorrect_pred = ([1 if x == 1 else 0 for x in answer_types ])
        bad_pred = ([1 if x == 4 else 0 for x in answer_types ])
    
        total = len(answer_types)
        
        output_dict["eval/correct_pred"] = np.sum(correct_pred)/total
        output_dict["eval/incorrect_pred"] = np.sum(incorrect_pred)/total
        output_dict["eval/bad_pred"] = np.sum(bad_pred)/total

        print(output_dict)

        np.save(os.path.join(config.model.model_path, "eval_correct_preds.npy"), correct_pred)
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

        for i_prompt, prompts in enumerate(eval_dataloader):
            outputs = model(input_ids= prompts["input_ids"], attention_mask = prompts["input_ids"]!=tokenizer.pad_token_id)
            logits  = outputs.logits[:, -1, A_to_D_tokens]
            logits = logits.softmax(dim=-1)
            A_to_D_logits_all.append(logits.tolist())

        A_to_D_logits_all = np.concatenate(A_to_D_logits_all, axis=0)
        np.save(os.path.join(config.model.model_path, "eval_A_to_D_probs.npy"), A_to_D_logits_all)

    

    dataset = load_dataset('sciq')['train']
    test_dataset = load_dataset('sciq')['validation']



    prompts_eval = list(map(prepare_prompt, list(test_dataset)))

    np.save(os.path.join(config.model.model_path, "eval_answers.npy"), [prompt_dict["answer"] for prompt_dict in prompts_eval])

    trainer = trlx.eval(
        eval_prompts=prompts_eval,
        # eval_prompts=prompts_train,
        metric_fn=metric_fn,
        eval_fn = eval_fn,
        config=config,
        stop_sequences = ["</s>"]
    )

if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

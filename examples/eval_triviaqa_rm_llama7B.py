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
import re
import string
import torch
import tqdm

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
    if (output == " True."):
        if answer == " True.":
            answer_type = 0
        else:
            answer_type = 1
    elif (output == " False."):
        if answer == " False.":
            answer_type = 2
        else:
            answer_type = 3
    else:
        answer_type = 4
    return answer_type


def prepare_prompt(question, answer, correct, split):
    prompt = {}
    prompt["prompt"] = "Q: "+question + " A: "+normalize_answer(answer)
    if correct:
        prompt["answer"] = " True."
    else:
        prompt["answer"] = " False."
    prompt["split"] = split
    return prompt


def main(hparams={}):
    model_path = "ckpts/rm_tiviaqa_llama7B_50_50/checkpoint_05000/hf_model/"

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
        np.save(os.path.join(config.model.model_path, "eval_True_False_samples_preds.npy"), np.array(kwargs["outputs"]))

        split_names = ["eval_false", "eval_true"]
        output_dict = {}

        for split_idx in range(len(split_names)):
            idxs = np.where(np.array(kwargs["split"])==split_idx)[0]
            
            answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[idxs], np.array(kwargs["answer"])[idxs]))
            
            
            true_positive = ([1 if x == 0 else 0 for x in answer_types ])
            false_positive = ([1 if x == 1 else 0 for x in answer_types ])
            true_negative = ([1 if x == 2 else 0 for x in answer_types ])
            false_negative = ([1 if x == 3 else 0  for x in answer_types])
            bad_pred = ([1 if x == 4 else 0 for x in answer_types ])
            total = len(answer_types)


            output_dict[split_names[split_idx]+"/true_positive"] = np.sum(true_positive)/total
            output_dict[split_names[split_idx]+"/false_positive"] = np.sum(false_positive)/total
            output_dict[split_names[split_idx]+"/true_negative"] = np.sum(true_negative)/total
            output_dict[split_names[split_idx]+"/false_negative"] = np.sum(false_negative)/total
            output_dict[split_names[split_idx]+"/bad_pred"] = np.sum(bad_pred)/total
            output_dict[split_names[split_idx]+"/accuracy"] = (np.sum(true_positive)+np.sum(true_negative))/total

        return output_dict



    def eval_fn(eval_dataloader, model, tokenizer, device, config, accelerator):
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
            log_likelihood = -loss.sum(axis=1)

            answer_log_probs_mean_all.append(log_likelihood.tolist())

        answer_log_probs_mean_all = np.concatenate(answer_log_probs_mean_all)
        np.save(os.path.join(config.model.model_path, "eval_False_log_prob_generation.npy"), answer_log_probs_mean_all)

    dataset_orig = load_dataset("trivia_qa", "unfiltered.nocontext")

    dataset = dataset_orig["train"]
    test_dataset = dataset_orig["validation"]

     
    questions = np.array(test_dataset["question"])
    true_facts = np.array([test_dataset[i]["answer"]["value"] for i in range(len(test_dataset))])
    false_facts_full = np.load("ckpts/sft_triviaqa_GPT2/checkpoint_05000/hf_model/eval_unfiltered_output_strings.npy")
    false_facts = []
    good_false_facts_idxs = []
    for i, fact in enumerate(false_facts_full):
        if "answer is" in fact and fact[-1] == ".":
            filtered_fact = fact.split("answer is")[1]
            if len(filtered_fact)>0 and filtered_fact[-1] == ".":
                filtered_fact = filtered_fact[:-1]
            filtered_fact = filtered_fact[1:]
            good_false_facts_idxs.append(i)
        else:
            filtered_fact = fact
            print(filtered_fact)
        false_facts.append(filtered_fact)


    false_facts = np.array(false_facts)
    idxs  = np.where(np.load("ckpts/sft_triviaqa_GPT2/checkpoint_05000/hf_model/eval_unfiltered_answers_correct.npy")==0)[0]
    idxs = np.intersect1d(idxs, good_false_facts_idxs)
    true_facts = true_facts[idxs]
    false_facts = false_facts[idxs]
    questions = questions[idxs]

    prompts_test_true = list(map(prepare_prompt, questions, true_facts , [1 for _ in range(len(questions))], [1 for _ in range(len(questions))]))
    prompts_test_false = list(map(prepare_prompt, questions, false_facts , [0 for _ in range(len(questions))], [0 for _ in range(len(questions))]))
    prompts_test = prompts_test_true + prompts_test_false
    # np.random.shuffle(prompts_test)

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

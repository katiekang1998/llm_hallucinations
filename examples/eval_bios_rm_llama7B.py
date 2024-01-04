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

import torch

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

def error_individial(output , answer) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]

    output = output[1: -1]
    try:
        prediction = int(output)
    except:
        return -1

    return abs(answer-prediction)


def prepare_prompt_correct(line, num_correct):
    prompt = {}
    prompt["prompt"] = line + " How many facts are true in the biography?"
    prompt["answer"] = num_correct
    return prompt

def prepare_prompt_total(line, num_total):
    prompt = {}
    prompt["prompt"] = line + " How many facts are in the biography total?"
    prompt["answer"] = num_total
    return prompt

def main(hparams={}):
    model_path = "ckpts/rm2_bios_llama7B_5/checkpoint_05000/hf_model/"

    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.model.model_path = model_path

    config.train.batch_size = 32

    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    
    config.method.gen_kwargs=dict(max_new_tokens=40, do_sample=False)


    # config.model.peft_config = LoraConfig(
    #     r=16,
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_alpha=16,
    #     lora_dropout=0, 
    # )

    def metric_fn(samples: List[str], **kwargs):
        output_dict = {}

        num_samples = len(samples)

        errors_correct = list(map(error_individial, np.array(kwargs["outputs"][:num_samples//2]), np.array(kwargs["answer"][:num_samples//2])))
        errors_correct = np.array(errors_correct)
        wrong_form_idxs = np.where(errors_correct<0)[0]
        correct_form_idxs = np.where(errors_correct>=0)[0]
        factuality_wrong_form_frac = len(wrong_form_idxs)/len(errors_correct)

        if len(correct_form_idxs)>0:
            factuality_avg_error = np.mean(errors_correct[correct_form_idxs])
        else:
            factuality_avg_error = -1

        errors_total = list(map(error_individial, np.array(kwargs["outputs"][num_samples//2:]), np.array(kwargs["answer"][num_samples//2:])))
        errors_total = np.array(errors_total)
        wrong_form_idxs = np.where(errors_total<0)[0]
        correct_form_idxs = np.where(errors_total>=0)[0]
        total_wrong_form_frac = len(wrong_form_idxs)/len(errors_total)

        if len(correct_form_idxs)>0:
            total_avg_error = np.mean(errors_total[correct_form_idxs])
        else:
            total_avg_error = -1

        output_dict["test/factuality_wrong_form_frac"] = float(factuality_wrong_form_frac)
        output_dict["test/factuality_avg_error"] = float(factuality_avg_error)
        output_dict["test/total_wrong_form_frac"] = float(total_wrong_form_frac)
        output_dict["test/total_avg_error"] = float(total_avg_error)

        return output_dict


    with open("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/factscores_test_medium.json", "r") as f:
        factscores = json.load(f)

    num_true_all = []
    num_total_all = []
    skipped_idxs = []
    for i in range(len(factscores["decisions"])):
        decison = factscores["decisions"][i]
        if decison == None:
            skipped_idxs.append(i)
        else:
            num_total_all.append(len(decison))
            num_true_all.append(np.sum([fact["is_supported"] for fact in decison]))

    generated_responses = np.load("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/output_strings_test_medium.npy")
    lines_all = []
    for i in range(len(generated_responses)):
        response = generated_responses[i]
        if i not in skipped_idxs:
            # if '<unk> ' in response:
            #     line = response.split('<unk> ')[1]
            line = response.split(': ')[1]
            lines_all.append(line)



    eval_type = "total"

    if eval_type == "factuality":
        test_prompts = list(map(prepare_prompt_correct, lines_all, num_true_all))
        save_file_name  = "test_medium_num_correct_zero_to_six_probs.npy"
    elif eval_type == "total":
        test_prompts = list(map(prepare_prompt_total, lines_all, num_total_all))
        save_file_name  = "test_medium_num_total_zero_to_six_probs.npy"



    def eval_fn(eval_dataloader, model, tokenizer, device, config):

        print("EVALUATING")

        zero_to_six_tokens = [29900, 29896, 29906, 29941, 29946, 29945, 29953]
        zero_to_six_logits_all = []

        for i_prompt, prompts in enumerate(eval_dataloader):
            samples = torch.cat([prompts["input_ids"], torch.Tensor([259]).int().to(device).repeat(prompts["input_ids"].shape[0], 1)], dim=1)
            outputs = model(input_ids= samples, attention_mask = samples!=tokenizer.pad_token_id)


            logits  = outputs.logits[:, -1, zero_to_six_tokens]
            logits = logits.softmax(dim=-1)
            zero_to_six_logits_all.append(logits.tolist())

        zero_to_six_logits_all = np.concatenate(zero_to_six_logits_all, axis=0)
        np.save(os.path.join(config.model.model_path, save_file_name), zero_to_six_logits_all)



    trainer = trlx.eval(
        eval_prompts=test_prompts,
        # eval_prompts=prompts_train,
        metric_fn=metric_fn,
        eval_fn = eval_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

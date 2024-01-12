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


def convert_output_to_int(output):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]

    output = output[1: -1]
    try:
        prediction = int(output)
    except:
        return -1

    return prediction


def prepare_prompt_correct(line, num_correct):
    prompt = {}
    prompt["prompt"] = line + " How many facts are true in the biography?"
    prompt["answer"] = num_correct
    return prompt

def prepare_sample_correct(line, num_correct):
    prompt = line + " How many facts are true in the biography?"
    response = " "+str(num_correct)+"."
    return (prompt, response)

def prepare_prompt_total(line, num_total):
    prompt = {}
    prompt["prompt"] = line + " How many facts are in the biography total?"
    prompt["answer"] = num_total
    return prompt

def prepare_sample_total(line, num_total):
    prompt = line + " How many facts are in the biography total?"
    response = " "+str(num_total)+"."
    return (prompt, response)

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 30000
    config.train.eval_interval = 500
    config.train.checkpoint_interval = 500
    config.train.batch_size=32
    config.train.checkpoint_dir = "ckpts/rm2_bios_llama7B_5_log_correct_ratio"
    # config.train.epochs = 100
    config.train.project_name = "trlx_rm2_bio2_llama7B"
    config.train.run_name = "train10000_5_log_correct_ratio"

    config.model.model_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=2.0e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
    config.scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e4, eta_min=1.0e-10)  # train.total_steps
        )

    config.model.peft_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0,
    )

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


        num_correct_predictions = np.array(list(map(convert_output_to_int, np.array(kwargs["outputs"][:num_samples//2]))))
        num_total_predictions = np.array(list(map(convert_output_to_int, np.array(kwargs["outputs"][num_samples//2:]))))

        correct_form_idxs = np.where((num_correct_predictions>=0)*(num_total_predictions>=0))[0]

        correct_ratio = np.array(kwargs["answer"][:num_samples//2])/np.array(kwargs["answer"][num_samples//2:])
        correct_ratio = correct_ratio[correct_form_idxs]

        num_total_predictions = np.clip(num_total_predictions, 0.01, 10)    
        correct_ratio_predictions = np.clip(num_correct_predictions[correct_form_idxs]/num_total_predictions[correct_form_idxs], 0, 1)

        correct_ratio_error = np.abs(correct_ratio-correct_ratio_predictions)

        output_dict["test/factuality_wrong_form_frac"] = float(factuality_wrong_form_frac)
        output_dict["test/factuality_avg_error"] = float(factuality_avg_error)
        output_dict["test/total_wrong_form_frac"] = float(total_wrong_form_frac)
        output_dict["test/total_avg_error"] = float(total_avg_error)
        output_dict["test/correct_ratio_error"] = float(np.mean(correct_ratio_error))
        
        return output_dict
    
    
    with open("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/factscores.json", "r") as f:
        factscores = json.load(f)
    
    with open("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/factscores_5000_10000.json", "r") as f:
        factscores2 = json.load(f)

    decisions = factscores["decisions"] + factscores2["decisions"]

    # decisions = factscores["decisions"]

    num_true_all = []
    num_total_all = []
    skipped_idxs = []
    for i in range(len(decisions)):
        decison = decisions[i]
        if decison == None:
            skipped_idxs.append(i)
        else:
            num_total = len(decison)
            num_total_all.append(min(num_total, 6))
            num_true = np.sum([fact["is_supported"] for fact in decison])
            num_true_all.append(min(num_true, 6))

    generated_responses = np.load("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/output_strings_train.npy")[:10000]
    lines_all = []
    for i in range(len(generated_responses)):
        response = generated_responses[i]
        if i not in skipped_idxs:
            # if '<unk> ' in response:
            #     line = response.split('<unk> ')[1]
            line = response.split(': ')[1]
            lines_all.append(line)

    train_samples = list(map(prepare_sample_correct, lines_all, num_true_all)) + list(map(prepare_sample_total, lines_all, num_total_all))
    np.random.shuffle(train_samples)


    with open("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/factscores_test_small.json", "r") as f:
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

    generated_responses = np.load("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/output_strings_test_small.npy")
    lines_all = []
    for i in range(len(generated_responses)):
        response = generated_responses[i]
        if i not in skipped_idxs:
            # if '<unk> ' in response:
            #     line = response.split('<unk> ')[1]
            line = response.split(': ')[1]
            lines_all.append(line)

    test_prompts = list(map(prepare_prompt_correct, lines_all, num_true_all)) + list(map(prepare_prompt_total, lines_all, num_total_all))

    trainer = trlx.train(
        samples=train_samples,
        eval_prompts=test_prompts,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

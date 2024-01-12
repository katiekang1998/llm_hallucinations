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

import pickle

def convert_output_to_int(output):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]

    try:
        output = output.lstrip()
        output = output.split(" ")
        num_correct_prediction = 0
        num_total_prediction = 0
        for i in range(len(output)):
            if output[i] == "True":
                num_correct_prediction += 1
            elif output[i] == "False":
                num_correct_prediction += 0
            else:
                return [-1, -1]
            num_total_prediction += 1
        return [num_total_prediction, num_correct_prediction]
    except:
        return [-1, -1]


def prepare_prompt(line, num_total, num_correct, split):
    prompt = {}
    prompt["prompt"] = line
    prompt["num_total"] = num_total
    prompt["num_correct"] = num_correct
    prompt["split"] = split
    return prompt

def prepare_sample(line, true_or_false):
    prompt = line
    response = ""
    for item in true_or_false:
        response += " "+ str(item)
    return (prompt, response)

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 30000
    # config.train.eval_interval = 500
    config.train.eval_interval = 100
    config.train.checkpoint_interval = 500
    config.train.batch_size=32
    config.train.checkpoint_dir = "ckpts/rm5_bios_llama7B"
    # config.train.epochs = 100
    config.train.project_name = "trlx_rm2_bio2_llama7B"
    config.train.run_name = "train10000_rm5"

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
        split_names = ["test", "True", "False"]
        output_dict = {}

        for split_idx in range(len(split_names)):
            idxs = np.where(np.array(kwargs["split"])==split_idx)[0]

            output_ints = np.array(list(map(convert_output_to_int, np.array(kwargs["outputs"])[idxs])))
            num_total_predictions = output_ints[:, 0]
            num_correct_predictions = output_ints[:, 1]

            correct_form_idxs = np.where((num_correct_predictions>=0)*(num_total_predictions>=0))[0]

            factuality_wrong_form_frac = np.sum(num_correct_predictions<0)/len(num_correct_predictions)
            total_wrong_form_frac = np.sum(num_total_predictions<0)/len(num_total_predictions)

            if len(correct_form_idxs)>0:
                correct_ratio = np.array(kwargs["num_correct"])[idxs][correct_form_idxs]/np.array(kwargs["num_total"])[idxs][correct_form_idxs]
                correct_ratio_predictions = np.clip(num_correct_predictions[correct_form_idxs]/num_total_predictions[correct_form_idxs], 0, 1)

                correct_ratio_error = np.abs(correct_ratio-correct_ratio_predictions)
                errors_correct = np.abs(np.array(kwargs["num_correct"])[idxs][correct_form_idxs]-num_correct_predictions[correct_form_idxs])
                errors_total = np.abs(np.array(kwargs["num_total"])[idxs][correct_form_idxs]-num_total_predictions[correct_form_idxs])
                factuality_avg_error = np.mean(errors_correct[errors_correct>=0])
                total_avg_error = np.mean(errors_total[errors_total>=0])
            else:
                factuality_avg_error = -1
                total_avg_error = -1
                correct_ratio_error = -1

            output_dict[split_names[split_idx]+"/factuality_wrong_form_frac"] = float(factuality_wrong_form_frac)
            output_dict[split_names[split_idx]+"/factuality_avg_error"] = float(factuality_avg_error)
            output_dict[split_names[split_idx]+"/total_wrong_form_frac"] = float(total_wrong_form_frac)
            output_dict[split_names[split_idx]+"/total_avg_error"] = float(total_avg_error)
            output_dict[split_names[split_idx]+"/correct_ratio_error"] = float(np.mean(correct_ratio_error))
        
        return output_dict
    
    
    with open("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/factscores.json", "r") as f:
        factscores = json.load(f)
    
    with open("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/factscores_5000_10000.json", "r") as f:
        factscores2 = json.load(f)

    decisions = factscores["decisions"] + factscores2["decisions"]

    # decisions = factscores["decisions"]

    true_or_false_all = []
    skipped_idxs = []
    for i in range(len(decisions)):
        decison = decisions[i]
        if decison == None:
            skipped_idxs.append(i)
        else:
            true_or_false = ([fact["is_supported"] for fact in decison])
            true_or_false_all.append(true_or_false)

    generated_responses = np.load("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/output_strings_train.npy")[:10000]
    lines_all = []
    for i in range(len(generated_responses)):
        response = generated_responses[i]
        if i not in skipped_idxs:
            # if '<unk> ' in response:
            #     line = response.split('<unk> ')[1]
            line = response.split(': ')[1]
            lines_all.append(line)

    train_samples = list(map(prepare_sample, lines_all, true_or_false_all))
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

    test_prompts_samples = list(map(prepare_prompt, lines_all, num_total_all, num_true_all, [0 for _ in range(len(lines_all))]))


    with open("biographies/test_bios_medium.pkl", "rb") as f:
        test_bios_medium = pickle.load(f)
    lines_all = test_bios_medium["bio"][:100]
    lines_all = list(map(lambda x: x.lstrip(), lines_all))
    num_true_all = [1 for _ in range(len(lines_all))]
    num_total_all = [1 for _ in range(len(lines_all))]

    test_prompts_True = list(map(prepare_prompt, lines_all, num_total_all, num_true_all, [1 for _ in range(len(lines_all))]))


    names = []
    bios = []

    for i in range(len(lines_all)):
        line = lines_all[i]
        if " is " in line:
            output = line.split(" is ")
            names.append(output[0])

            bio = ""
            for j in range(1, len(output)):
                bio+= " is " + output[j]
            bios.append(bio)

        elif " was " in line:
            output = line.split(" was ")
            names.append(output[0])

            bio = ""
            for j in range(1, len(output)):
                bio+= " was " + output[j]
            bios.append(bio)

    lines_all = []
    rand_idxs = np.random.choice(len(names), len(names), replace=False)
    for i in range(len(names)):
        lines_all.append(names[rand_idxs[i]]+bios[i])
    
    num_true_all = [0 for _ in range(len(lines_all))]

    test_prompts_False = list(map(prepare_prompt, lines_all, num_total_all, num_true_all, [2 for _ in range(len(lines_all))]))

    test_prompts = test_prompts_samples + test_prompts_True + test_prompts_False


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

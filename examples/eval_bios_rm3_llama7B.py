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
import pickle
import torch

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

def convert_output_to_int(output):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]

    output = output[1:]

    if len(output) != 2:
        return [-1, -1]
    try:
        num_total_prediction = int(output[0])
        num_correct_prediction = int(output[1])
    except:
        return [-1, -1]

    return [num_total_prediction, num_correct_prediction]


def prepare_prompt(line, num_total, num_correct):
    prompt = {}
    prompt["prompt"] = line
    prompt["num_total"] = num_total
    prompt["num_correct"] = num_correct
    return prompt

def prepare_sample(line, num_total, num_correct):
    prompt = line
    response = " "+str(num_total)+str(num_correct)
    return (prompt, response)


def main(hparams={}):
    model_path = "ckpts/rm3_bios_llama7B/checkpoint_30000/hf_model/"

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

        output_ints = np.array(list(map(convert_output_to_int, np.array(kwargs["outputs"]))))
        num_total_predictions = output_ints[:, 0]
        num_correct_predictions = output_ints[:, 1]

        np.save(os.path.join(config.model.model_path, "test_medium_True_num_total_preds.npy"), num_total_predictions)
        np.save(os.path.join(config.model.model_path, "test_medium_True_num_correct_preds.npy"), num_correct_predictions)

        correct_form_idxs = np.where((num_correct_predictions>=0)*(num_total_predictions>=0))[0]

        factuality_wrong_form_frac = np.sum(num_correct_predictions<0)/len(num_correct_predictions)
        total_wrong_form_frac = np.sum(num_total_predictions<0)/len(num_total_predictions)

        if len(correct_form_idxs)>0:
            correct_ratio = np.array(kwargs["num_correct"])[correct_form_idxs]/np.array(kwargs["num_total"])[correct_form_idxs]
            correct_ratio_predictions = np.clip(num_correct_predictions[correct_form_idxs]/num_total_predictions[correct_form_idxs], 0, 1)

            correct_ratio_error = np.abs(correct_ratio-correct_ratio_predictions)
            errors_correct = np.abs(np.array(kwargs["num_correct"])[correct_form_idxs]-num_correct_predictions[correct_form_idxs])
            errors_total = np.abs(np.array(kwargs["num_total"])[correct_form_idxs]-num_total_predictions[correct_form_idxs])
            factuality_avg_error = np.mean(errors_correct[errors_correct>=0])
            total_avg_error = np.mean(errors_total[errors_total>=0])
        else:
            factuality_avg_error = -1
            total_avg_error = -1
            correct_ratio_error = -1

        output_dict["test/factuality_wrong_form_frac"] = float(factuality_wrong_form_frac)
        output_dict["test/factuality_avg_error"] = float(factuality_avg_error)
        output_dict["test/total_wrong_form_frac"] = float(total_wrong_form_frac)
        output_dict["test/total_avg_error"] = float(total_avg_error)
        output_dict["test/correct_ratio_error"] = float(np.mean(correct_ratio_error))
        
        return output_dict

    # with open("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/factscores_test_medium.json", "r") as f:
    #     factscores = json.load(f)

    # num_true_all = []
    # num_total_all = []
    # skipped_idxs = []
    # for i in range(len(factscores["decisions"])):
    #     decison = factscores["decisions"][i]
    #     if decison == None:
    #         skipped_idxs.append(i)
    #     else:
    #         num_total_all.append(len(decison))
    #         num_true_all.append(np.sum([fact["is_supported"] for fact in decison]))

    # generated_responses = np.load("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/output_strings_test_medium.npy")
    # lines_all = []
    # for i in range(len(generated_responses)):
    #     response = generated_responses[i]
    #     if i not in skipped_idxs:
    #         # if '<unk> ' in response:
    #         #     line = response.split('<unk> ')[1]
    #         line = response.split(': ')[1]
    #         lines_all.append(line)

    with open("biographies/test_bios_medium.pkl", "rb") as f:
        test_bios_medium = pickle.load(f)
    lines_all = test_bios_medium["bio"]
    lines_all = list(map(lambda x: x.lstrip(), lines_all))
    num_true_all = [-1 for _ in range(len(lines_all))]
    num_total_all = [-1 for _ in range(len(lines_all))]


    # names = []
    # bios = []

    # for i in range(len(lines_all)):
    #     line = lines_all[i]
    #     if " is " in line:
    #         output = line.split(" is ")
    #         names.append(output[0])

    #         bio = ""
    #         for j in range(1, len(output)):
    #             bio+= " is " + output[j]
    #         bios.append(bio)

    #     elif " was " in line:
    #         output = line.split(" was ")
    #         names.append(output[0])

    #         bio = ""
    #         for j in range(1, len(output)):
    #             bio+= " was " + output[j]
    #         bios.append(bio)

    # lines_all = []
    # rand_idxs = np.random.choice(len(names), len(names), replace=False)
    # for i in range(len(names)):
    #     lines_all.append(names[rand_idxs[i]]+bios[i])

    test_prompts = list(map(prepare_prompt, lines_all, num_total_all, num_true_all))


    # def eval_fn(eval_dataloader, model, tokenizer, device, config):

    #     print("EVALUATING")

    #     zero_to_six_tokens = [29900, 29896, 29906, 29941, 29946, 29945, 29953]
    #     zero_to_six_logits_all = []

    #     for i_prompt, prompts in enumerate(eval_dataloader):
    #         samples = torch.cat([prompts["input_ids"], torch.Tensor([259]).int().to(device).repeat(prompts["input_ids"].shape[0], 1)], dim=1)
    #         outputs = model(input_ids= samples, attention_mask = samples!=tokenizer.pad_token_id)


    #         logits  = outputs.logits[:, -1, zero_to_six_tokens]
    #         logits = logits.softmax(dim=-1)
    #         zero_to_six_logits_all.append(logits.tolist())

    #     zero_to_six_logits_all = np.concatenate(zero_to_six_logits_all, axis=0)
    #     np.save(os.path.join(config.model.model_path, save_file_name), zero_to_six_logits_all)



    trainer = trlx.eval(
        eval_prompts=test_prompts,
        # eval_prompts=prompts_train,
        metric_fn=metric_fn,
        eval_fn = None,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

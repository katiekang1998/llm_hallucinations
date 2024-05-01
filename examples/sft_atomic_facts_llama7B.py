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


def prepare_prompt(line, num_total, num_correct):
    prompt = {}
    prompt["prompt"] = line
    prompt["num_total"] = num_total
    prompt["num_correct"] = num_correct
    return prompt

def prepare_sample(line, facts, num_total, num_correct):
    prompt = line
    response = " "+facts
    return (prompt, response)

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 30000
    # config.train.eval_interval = 500
    config.train.eval_interval = 500
    config.train.checkpoint_interval = 500
    config.train.batch_size=32//2
    config.train.checkpoint_dir = "ckpts/sft_atomic_facts_llama7B"
    # config.train.epochs = 100
    config.train.project_name = "trlx_sft_atomic_facts_llama7B"
    config.train.run_name = "train10000"
    config.train.num_log_samples = -1 


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
    config.method.gen_kwargs=dict(
                max_new_tokens=500,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            )

    def metric_fn(samples: List[str], **kwargs):
        
        return {}
    
    
    with open("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/factscores.json", "r") as f:
        factscores = json.load(f)
    
    with open("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/factscores_5000_10000.json", "r") as f:
        factscores2 = json.load(f)

    decisions = factscores["decisions"] + factscores2["decisions"]

    # decisions = factscores["decisions"]

    num_true_all = []
    num_total_all = []
    facts_all = []
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
            facts_string = ""
            for j in range(len(decison)):
                facts_string += "Fact "+str(j+1)+": "
                facts_string += decison[j]["atom"] + "\n" 
            facts_all.append(facts_string)
    
                

    generated_responses = np.load("ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/output_strings_train.npy")[:10000]
    lines_all = []
    for i in range(len(generated_responses)):
        response = generated_responses[i]
        if i not in skipped_idxs:
            # if '<unk> ' in response:
            #     line = response.split('<unk> ')[1]
            line = response.split(': ')[1]
            lines_all.append(line)

    train_samples = list(map(prepare_sample, lines_all, facts_all, num_total_all, num_true_all))
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

    test_prompts = list(map(prepare_prompt, lines_all, num_total_all, num_true_all))

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

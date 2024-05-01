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



def prepare_prompt(line,idx):
    prompt = {}
    prompt["prompt"] = line
    prompt["idx"] = idx
    return prompt


def main(hparams={}):
    model_path = "ckpts/sft_atomic_facts_llama7B/checkpoint_01000/hf_model/"

    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.model.model_path = model_path

    config.train.batch_size = 32

    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    
    config.method.gen_kwargs=dict(max_new_tokens=200, do_sample=False)


    # config.model.peft_config = LoraConfig(
    #     r=16,
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_alpha=16,
    #     lora_dropout=0, 
    # )

    def metric_fn(samples: List[str], **kwargs):

        import IPython; IPython.embed()

        facts_all = []
        facts_idxs = []
        for i, row in enumerate(kwargs["outputs"]):
            facts = (row.split("\n"))
            for fact in facts:
                try:
                    prefix = fact.split(": ")[0]
                    fact = fact[len(prefix)+2:]
                    facts_all.append(fact)
                    facts_idxs.append(kwargs["idx"][i])
                except:
                    print(i)
                    print(fact)
        
        save_dict = {}
        save_dict["facts"] = facts_all
        save_dict["bio_idxs"] = facts_idxs

        np.save(os.path.join(config.model.model_path, "test_medium_False_facts.npy"), save_dict)        
        return {}

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


    names = []
    bios = []
    idxs = []

    for i in range(len(lines_all)):
        line = lines_all[i]
        if " is " in line:
            output = line.split(" is ")
            names.append(output[0])

            bio = ""
            for j in range(1, len(output)):
                bio+= " is " + output[j]
            bios.append(bio)
            idxs.append(i)

        elif " was " in line:
            output = line.split(" was ")
            names.append(output[0])

            bio = ""
            for j in range(1, len(output)):
                bio+= " was " + output[j]
            bios.append(bio)
            idxs.append(i)
        # else:
        #     print(line)

    lines_all = []
    rand_idxs = np.random.choice(len(bios), len(bios), replace=False)
    for i in range(len(names)):
        lines_all.append(names[i]+bios[rand_idxs[i]])


    test_prompts = list(map(prepare_prompt, lines_all, idxs))


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

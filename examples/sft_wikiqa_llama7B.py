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
from factscore.factscorer import FactScorer

import pickle


from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)



def prepare_sample(question, answer):
    question = question.lower()
    return ("Question: "+ question, " Answer: " + answer)

def prepare_prompt(question, document_title):
    question = question.lower()
    prompt = {}
    prompt["prompt"] = "Question: "+ question
    prompt["document_title"] = document_title
    return prompt

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 30000
    config.train.eval_interval = 500
    # config.train.eval_interval = 50

    config.train.checkpoint_interval = 500
    config.train.checkpoint_dir = "ckpts/sft_wikiqa_llama7B"
    # config.train.epochs = 100
    config.train.project_name = "trlx_sft_wikiqa_llama7B"
    config.train.run_name = "orig"
    config.train.num_log_samples = 5
    config.train.batch_size = 8

    config.model.model_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1.0e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
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

    config.method.gen_kwargs=dict(max_new_tokens=200, top_k=0, top_p=1.0, do_sample=True)
    fs = FactScorer(openai_key="/data/katie_kang/openai_key_file.txt", data_dir="/data/katie_kang/trlx/examples/.cache/factscore", cache_dir="/data/katie_kang/trlx/examples/.cache/factscore")

    def metric_fn(samples: List[str], **kwargs):
        names = kwargs["document_title"]
        outputs = kwargs["outputs"]

        good_idxs = []
        bad_idxs = []
        good_outputs = []
        for i in range(len(outputs)):
            output = outputs[i]
            if output[-len(" </s>"):] == " </s>":
                output = output[: -len(" </s>")]
            if output[-len("</s>"):] == "</s>":
                output = output[: -len("</s>")]

            if output[: len(" Answer: ")] == " Answer: " and output[-1] == ".":
                good_outputs.append(output[len(" Answer: "):])
                good_idxs.append(i)
            else:
                bad_idxs.append(i)

        output_dict = {}
        
        if len(good_idxs) >0:

            factscores = fs.get_score(list(np.array(names)[good_idxs]), good_outputs, gamma=0)

            num_true_all = []
            num_total_all = []
            frac_correct_facts = []
            num_none_decisions = 0
            for i in range(len(factscores["decisions"])):
                decison = factscores["decisions"][i]
                if decison == None:
                    num_total_all.append(0)
                    num_true_all.append(0)
                    print(good_outputs[i])
                    num_none_decisions += 1

                else:
                    num_total_all.append(len(decison))
                    num_true_all.append(np.sum([fact["is_supported"] for fact in decison]))
                    frac_correct_facts.append(np.sum([fact["is_supported"] for fact in decison])/len(decison))
            num_total_all = np.array(num_total_all)
            num_true_all = np.array(num_true_all)
            frac_correct_facts = np.array(frac_correct_facts)

            if len(num_total_all)>0:
                output_dict["test/avg_num_facts"] = np.mean(num_total_all)
                output_dict["test/avg_num_correct_facts"] = np.mean(num_true_all)
                output_dict["test/avg_num_false_facts"] =   np.mean(num_total_all - num_true_all)
                output_dict["test/avg_frac_correct_facts"] = np.mean(frac_correct_facts)
            output_dict["test/frac_wrong"] = (len(bad_idxs)+num_none_decisions) / len(outputs)

        else:
            output_dict["test/frac_wrong"] = len(bad_idxs) / len(outputs)

        return output_dict
    
    data = load_dataset("wiki_qa")

    train_data = data["train"].select(np.where(np.array(data["train"]["label"])==1)[0])

    test_data = data["validation"].select(np.where(np.array(data["validation"]["label"])==1)[0])

    bad_contexts = ['Pitbull (entertainer)',
        'List of Xbox 360 games',
        'Federal Reserve System',
        'Plus-minus sign',
        'Plus-minus sign',
        'ShopNBC',
        'Thyroid hormone',
        'Stand by Me (song)',
        'ARB',
        'Newton metre',
        'Skins (UK TV series)',
        'Triumph Motorcycles',
        'Triumph Motorcycles',
        'Olmec',
        'List of Arizona state symbols',
        'List of Arizona state symbols',
        'Tiger Salamander',
        'Arctic circle',
        'GE Building',
        'JÃ¤gerbomb',
        'List of National Basketball Association season scoring leaders',
        'Direct Marketing']

    test_data = test_data.select(np.where([document_title not in bad_contexts for document_title in test_data["document_title"]])[0])

    train_samples = list(map(prepare_sample, train_data["question"], train_data["answer"]))

    prompts_test = list(map(prepare_prompt,test_data["question"], test_data["document_title"]))

    np.random.shuffle(train_samples)
    

    trainer = trlx.train(
        samples=train_samples,
        eval_prompts=prompts_test,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

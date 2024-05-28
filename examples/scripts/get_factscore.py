import json
from factscore.factscorer import FactScorer
import numpy as np
import pickle
import os


# checkpoint = "../ckpts/ppo_wikibios_true2_false-3_rm_llama7B/checkpoint_032000/hf_model"
checkpoint = "../ckpts/ppo_wikibios_true2_false-3_rm_gpt3pt5/checkpoint_010000/hf_model"


samples = np.load(checkpoint+"/sample_output_strings_test.npy")
names = np.load("../ckpts/wikibios_data/test_names.npy")

generations = []

for sample in samples:
    generations.append(sample.split("Bio: ")[1].lstrip())

topics = names

# import IPython; IPython.embed()

fs = FactScorer(openai_key="/data/katie_kang/openai_key_file.txt", data_dir="/data/katie_kang/trlx/examples/.cache/factscore", cache_dir="/data/katie_kang/trlx/examples/.cache/factscore1")
# fs = FactScorer(openai_key="/data/katie_kang/openai_key_file_rail.txt", data_dir="/data/katie_kang/trlx/examples/.cache/factscore", cache_dir="/data/katie_kang/trlx/examples/.cache/factscore2")

out = fs.get_score(list(topics), list(generations), gamma=0)


with open(checkpoint+"/sample_output_strings_test_factscore.json", "w") as f:
    json.dump(out, f)
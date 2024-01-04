import json
from factscore.factscorer import FactScorer
import numpy as np

# with open("/data/katie_kang/trlx/examples/wandb/run-20231212_230917-3jf0ox3w/files/media/table/samples_14999_9cecc0d9d35c8902efe7.table.json") as f:
# with open("/data/katie_kang/trlx/examples/wandb/run-20231215_001613-7wwyox3f/files/media/table/samples_9999_1033c10350ec912d378d.table.json") as f:
#     bios = json.load(f)

# topics = [prompt.split("biography for ")[1][:-1] for prompt in np.array(bios["data"])[:, 0]]
# generations = np.array(bios["data"])[:, 1]
# generations = [generation[1:] for generation in generations]



train_generations  = np.load("../ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/output_strings_test_medium.npy")

print(len(train_generations))
topics = [sample.split("biography for ")[1].split(":")[0] for sample in np.array(train_generations)]

generations = []
for sample in train_generations:
    prompt_len = len(sample.split(":")[0])
    generations.append(sample[prompt_len+1:].lstrip())


fs = FactScorer(openai_key="/data/katie_kang/openai_key_file.txt", data_dir="/data/katie_kang/trlx/examples/.cache/factscore", cache_dir="/data/katie_kang/trlx/examples/.cache/factscore")
out = fs.get_score(topics, generations, gamma=0)

with open("../ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/factscores_test_medium.json", "w") as f:
    json.dump(out, f)
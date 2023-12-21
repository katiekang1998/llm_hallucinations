import json
from factscore.factscorer import FactScorer
import numpy as np

# with open("/data/katie_kang/trlx/examples/wandb/run-20231212_230917-3jf0ox3w/files/media/table/samples_14999_9cecc0d9d35c8902efe7.table.json") as f:
with open("/data/katie_kang/trlx/examples/wandb/run-20231215_001613-7wwyox3f/files/media/table/samples_4999_226e42fada1d25bf9f7b.table.json") as f:
    bios = json.load(f)

topics = [prompt.split("biography for ")[1][:-1] for prompt in np.array(bios["data"])[:, 0]]
generations = np.array(bios["data"])[:, 1]
generations = [generation[1:] for generation in generations]


fs = FactScorer(openai_key="/data/katie_kang/openai_key_file.txt", data_dir="/data/katie_kang/trlx/examples/.cache/factscore", cache_dir="/data/katie_kang/trlx/examples/.cache/factscore")
out = fs.get_score(topics, generations)


np.save("../ckpts/sft_bios_new_llama7B/checkpoint_05000/hf_model/topics.npy", topics)
np.save("../ckpts/sft_bios_new_llama7B/checkpoint_05000/hf_model/generations.npy", generations)
# np.save("../ckpts/sft_bios_llama7B/checkpoint_00500/hf_model/factscores.npy", out)

with open("../ckpts/sft_bios_new_llama7B/checkpoint_05000/hf_model/factscores.json", "w") as f:
    json.dump(out, f)
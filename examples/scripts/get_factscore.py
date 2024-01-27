import json
from factscore.factscorer import FactScorer
import numpy as np
import pickle
import os

# with open("/data/katie_kang/trlx/examples/wandb/run-20231212_230917-3jf0ox3w/files/media/table/samples_14999_9cecc0d9d35c8902efe7.table.json") as f:
# with open("/data/katie_kang/trlx/examples/wandb/run-20231215_001613-7wwyox3f/files/media/table/samples_9999_1033c10350ec912d378d.table.json") as f:
#     bios = json.load(f)

# topics = [prompt.split("biography for ")[1][:-1] for prompt in np.array(bios["data"])[:, 0]]
# generations = np.array(bios["data"])[:, 1]
# generations = [generation[1:] for generation in generations]



# train_generations  = np.load("../ckpts/sft_bios_new_llama7B/checkpoint_20000/hf_model/output_strings_train.npy")

# print(len(train_generations))
# topics = [sample.split("biography for ")[1].split(":")[0] for sample in np.array(train_generations)]

# generations = []
# for sample in train_generations:
#     prompt_len = len(sample.split(":")[0])
#     generations.append(sample[prompt_len+1:].lstrip())



# with open('../biographies/train_bios.pkl', 'rb') as fp:
#     train_data = pickle.load(fp)

# names = train_data["name"][:100]
# lines_all = train_data["bio"][:100]
# lines_all = list(map(lambda x: x.lstrip(), lines_all))

# names2 = []
# bios = []

# for i in range(len(lines_all)):
#     line = lines_all[i]
#     if " is " in line:
#         output = line.split(" is ")
#         names2.append(output[0])

#         bio = ""
#         for j in range(1, len(output)):
#             bio+= " is " + output[j]
#         bios.append(" is "+bio)

#     elif " was " in line:
#         output = line.split(" was ")
#         names2.append(output[0])

#         bio = ""
#         for j in range(1, len(output)):
#             bio+= " was " + output[j]
#         bios.append(" was "+bio)

# lines_all = []
# names3 = []
# rand_idxs = np.random.choice(len(names2), len(names2), replace=False)
    
# for i in range(len(names2)):
#     names3.append(names[rand_idxs[i]])
#     lines_all.append(names2[rand_idxs[i]]+bios[i])




# with open("../biographies/test_bios_medium.pkl", "rb") as f:
#     train_bios = pickle.load(f)
# topics = train_bios["name"]

# generations = []

# for bio in train_bios["bio"]:
#     generations.append(bio.lstrip())


generations = []
for bio in np.load("../movies/train_plots_gpt3pt5.npy"):
    generations.append(bio.lstrip())

titles = []
with open("/data/katie_kang/trlx/examples/movies/titles",) as file:
    for line in file:
        titles.append(line.strip())
titles = np.array(titles)
train_idxs = np.load("/data/katie_kang/trlx/examples/movies/common_train_idxs.npy")

topics = titles[train_idxs]

# generations = []
# for bio in np.load("../biographies/test_bios_gpt3pt5_small.npy"):
#     generations.append(bio.lstrip())

import IPython; IPython.embed()

fs = FactScorer(openai_key="/data/katie_kang/openai_key_file_rail.txt", data_dir="/data/katie_kang/trlx/examples/.cache/factscore", cache_dir="/data/katie_kang/trlx/examples/.cache/factscore")
out = fs.get_score(list(topics), list(generations), gamma=0)


# with open("../biographies/factscores_train10000_gpt3pt5.json", "w") as f:
#     json.dump(out, f)

with open("../movies/factscores_train_gpt3pt5.json", "w") as f:
    json.dump(out, f)





# # # samples = np.load(os.path.join("../ckpts/ppo_rm_bios_llama7B_true2_false-3_kl0pt5_GPT3pt5/checkpoint_006000/hf_model", "sample_output_strings_test_medium.npy"))
# samples = np.load(os.path.join("../ckpts/ppo_rm_wikiplots_llama7B_true2_false-3_kl0pt5/checkpoint_030000/hf_model", "sample_output_strings_eval_medium.npy"))

# generations = []

# for sample in samples:
#     # if "Bio: "  in sample:
#     if "Answer: "  in sample:
#         generations.append(sample.split("Answer: ")[1].lstrip())
#         # generations.append(sample.split("Bio: ")[1].lstrip())
#     else:
#         print(sample)
#         generations.append(sample.lstrip())

# # names = np.load("../biographies/names.npy")
# # test_idxs = np.load("../biographies/test_points_medium.npy")
# # topics = names[test_idxs]

# titles = []
# with open("/data/katie_kang/trlx/examples/movies/titles",) as file:
#     for line in file:
#         titles.append(line.strip())

# train_idxs = np.load("/data/katie_kang/trlx/examples/movies/common_train_idxs.npy")
# test_idxs = np.load("/data/katie_kang/trlx/examples/movies/common_test_medium_idxs.npy")

# topics = np.array(titles)[test_idxs]
# import IPython; IPython.embed()

# fs = FactScorer(openai_key="/data/katie_kang/openai_key_file.txt", data_dir="/data/katie_kang/trlx/examples/.cache/factscore", cache_dir="/data/katie_kang/trlx/examples/.cache/factscore")
# out = fs.get_score(list(topics), list(generations), gamma=0)

# with open("../ckpts/ppo_rm_wikiplots_llama7B_true2_false-3_kl0pt5/checkpoint_030000/hf_model/factscores_eval.json", "w") as f:
#     json.dump(out, f)


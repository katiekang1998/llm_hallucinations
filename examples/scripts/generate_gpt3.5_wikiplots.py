import openai
import re
import string
import numpy as np
from tqdm import tqdm
import time
import pickle
import tqdm


def call_instructgpt_with_names(names):
    # if exact_match_score(model_prediction, ground_truth):
    #     return 1
    api_key = open("/data/katie_kang/openai_key_file.txt", "r").read()
    openai.api_key = api_key.strip()

    prompt_template1 = """Generate a one sentence plot summary for all 8 of the following films/books/TV series: 

Title 1: Animal Farm
Title 2: The Line of Freedom (film)
Title 3: Duke Nukem 3D
""" 


    prompt_template2 = ""

    for i in range(len(names)):
        prompt_template2 += """Title {}: {}
""".format(i+4, names[i])

    prompt_template3 = """
Plot 1: Old Major, the old boar on the Manor Farm, summons the animals on the farm together for a meeting, during which he refers to humans as "enemies" and teaches the animals a revolutionary song called "Beasts of England".
Plot 2: The story follows Nasir Baloch, a young student rights activist who is abducted and tortured by his countries security forces.
Plot 3: Duke Nukem 3D is set on Earth sometime in the early 21st century.
"""

    prompt = prompt_template1 + prompt_template2 + prompt_template3

    num_tries = 0

    while True:
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",  # or another model version
                # prompt=[filled_prompt, filled_prompt],
                prompt=prompt,
                max_tokens=300,
                temperature=0.0,
            )

            response = response.choices[0].text.strip()
            bios = response.split("\n")
            assert(len(bios) == len(names))
            bios_filtered = []
            for bio in bios:
                prefix = bio.split(": ")[0]
                bio = bio[len(prefix)+2:]
                bios_filtered.append(" " +bio)
            return bios_filtered
        except:
            import IPython; IPython.embed()
            print("EXCEPTION", num_tries)
            time.sleep(3)
            num_tries+=1



titles = []
with open("/data/katie_kang/trlx/examples/movies/titles",) as file:
    for line in file:
        titles.append(line.strip())
titles = np.array(titles)
train_idxs = np.load("/data/katie_kang/trlx/examples/movies/common_train_idxs.npy")

train_title_names = titles[train_idxs]

# with open("../biographies/train_bios.pkl", "rb") as f:
#     train_bios = pickle.load(f)
# train_bios_names = train_bios["name"][:10000]

bios_all = np.load("../movies/train_plots_gpt3pt5.npy").tolist()
for i in tqdm.tqdm(range(9850//5, len(train_title_names)//5)):
    names = train_title_names[5*i:5*(i+1)]
    bios = call_instructgpt_with_names(names)
    bios_all.extend(bios)

import IPython; IPython.embed()

np.save("../movies/train_plots_gpt3pt5.npy", bios_all)
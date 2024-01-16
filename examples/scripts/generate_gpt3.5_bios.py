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

    prompt_template1 = """Generate a one line biography for the following people. 

Name 1: Velusami Radhakrishnan
Name 2: Andrew Taylor (architect)
Name 3: Diego Herner
""" 


    prompt_template2 = ""

    for i in range(len(names)):
        prompt_template2 += """Name {}: {}
""".format(i+4, names[i])

    prompt_template3 = """
Bio 1: Velusami Radhakrishnan is a Sri Lankan politician and state minister.
Bio 2: Andrew Taylor was a British architect and councillor.
Bio 3: Diego Herner is a retired Argentine footballer, who played as a centre-back."""

    prompt = prompt_template1 + prompt_template2 + prompt_template3

    num_tries = 0

    while True:
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",  # or another model version
                # prompt=[filled_prompt, filled_prompt],
                prompt=prompt,
                max_tokens=200,
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
            print("EXCEPTION", num_tries)
            time.sleep(3)
            num_tries+=1



names = np.load("../biographies/names.npy")

test_idxs = np.load("../biographies/test_points_small.npy")

test_bios_names = names[test_idxs]

# with open("../biographies/train_bios.pkl", "rb") as f:
#     train_bios = pickle.load(f)
# train_bios_names = train_bios["name"][:10000]

bios_all = []
for i in tqdm.tqdm(range(len(test_bios_names)//5)):
    names = test_bios_names[5*i:5*(i+1)]
    bios = call_instructgpt_with_names(names)
    bios_all.extend(bios)

import IPython; IPython.embed()

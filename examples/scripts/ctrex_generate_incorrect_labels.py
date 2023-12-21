import json
import numpy as np
from datasets import load_dataset
import editdistance
from tqdm import tqdm

#load json
with open('../trex_relations2questions2_cleaned.json', 'rb') as fp:
    template2question = json.load(fp)

with open('../trex_relations2idxs.json', 'rb') as fp:
    trex_relations2idxs = json.load(fp)

def convert_template_to_question(template_sub_label):
    template, sub_label = template_sub_label
    assert(template in template2question.keys())
    question = template2question[template][1]
    question = question.replace("[X]", sub_label)
    return question


dataset_orig = load_dataset('relbert/t_rex')
train_idxs = np.load("../custom_trex/test_points_small.npy")

dataset = dataset_orig["train"].select(train_idxs)
train_relations = dataset["relation"]
train_heads = dataset["head"]
train_tails = dataset["tail"]
orig_dataset_tails = dataset_orig["train"]["tail"]

incorrect_answers_all = []

for i in tqdm(range(len(train_relations))):

    rand_idx = i
    question = convert_template_to_question((train_relations[rand_idx], train_heads[rand_idx]))
    answer = train_tails[rand_idx]
    incorrect_answer = orig_dataset_tails[np.random.choice(trex_relations2idxs[train_relations[rand_idx]])]
    if incorrect_answer == answer:
        incorrect_answer = np.random.choice(orig_dataset_tails)
    incorrect_answers_all.append(incorrect_answer)
    # if editdistance.eval(answer, incorrect_answer)/len(answer)<0.2:
    #     print(question)
    #     print("answer1: ", answer)
    #     print("answer2: ", incorrect_answer)
    #     print("edit distance: ", editdistance.eval(answer, incorrect_answer))
    #     print("edit distance ratio: ", editdistance.eval(answer, incorrect_answer)/len(answer))
    #     print("")

np.save("../custom_trex/test_small_incorrect_tails.npy", np.array(incorrect_answers_all))
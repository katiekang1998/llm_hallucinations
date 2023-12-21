from datasets import load_dataset
import sqlite3
import numpy as np
import tqdm
import wikipediaapi
from concurrent import futures
import pickle
import time

names = np.load("../biographies/names.npy")

wiki_wiki = wikipediaapi.Wikipedia('KatieKang (katiekang1998@berkeley.edu)', 'en')

train_idxs = np.load("../biographies/train_points.npy")

with open("../biographies/train_bios.pkl", "rb") as f:
    old_train_dict = pickle.load(f)


def get_bio(name):
    page_py = wiki_wiki.page(name)
    #see if page is valid
    # try:
    try:
        if page_py.exists() == False:
            return ""
    except:
        print(name)
        return ""

    wiki_summary = page_py.summary
    wiki_summary_split = wiki_summary.split(".")
    wiki_summary_first_sentence = wiki_summary_split[0]
    response = " " + wiki_summary_first_sentence + "."
    split_idx = 0
    if len(response) < 3:
        return ""
    while response[-3] == " ":
        split_idx += 1
        if len(wiki_summary_split) <= split_idx:
            break
        else:
            response = response + wiki_summary_split[split_idx] + "."
    return response
    # except:
    #     print(name)
    #     return ""

executor = futures.ThreadPoolExecutor()

train_names = old_train_dict["name"]
train_bios = old_train_dict["bio"]
for i in tqdm.tqdm(range((len(train_names)//10)+1, (len(train_idxs)//10)+1)):

    name_idx_batch = train_idxs[i*10:(i+1)*10]
    names_batch = names[name_idx_batch]
    responses_batch = executor.map(get_bio, names_batch)
    responses_batch = list(responses_batch)

    for batch_idx, response in enumerate(responses_batch):
        name = names_batch[batch_idx]
        if len(response) > 0:
            train_names.append(name)
            train_bios.append(response)

    if i%1000 == 0:
        print("SAVING", i)
        train_dict = {"name": train_names, "bio": train_bios}

        with open("../biographies/train_bios.pkl", "wb") as f:
            pickle.dump(train_dict, f)

# def is_name_in_db(con, title):
#     # creating cursor
#     cursor = con.cursor()
#     cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
#     results = cursor.fetchall()
#     if len(results) > 0:
#         return True
#     else:
#         return False

# # creating file path
# dbfile = '/data/katie_kang/trlx/examples/.cache/factscore/enwiki-20230401.db'
# # Create a SQL connection to our SQLite database
# con = sqlite3.connect(dbfile)

# bios_data = load_dataset('wiki_bio')


# good_contexts = []
# for i in tqdm.tqdm(range(len(bios_data["train"]))):
#     context = bios_data["train"][i]['input_text']['context']
#     context = context.strip().replace("-lrb- ", "(").replace(" -rrb-", ")")
#     split_context = context.split("(")
#     if len(split_context) > 1:
#         context = split_context[0].title() + "(" + split_context[1].lower()
        
#     else:
#         context = context.title()
#     if is_name_in_db(con, context):
#         good_contexts.append(context)

# for i in tqdm.tqdm(range(len(bios_data["test"]))):
#     context = bios_data["test"][i]['input_text']['context']
#     context = context.strip().replace("-lrb- ", "(").replace(" -rrb-", ")")
#     split_context = context.split("(")
#     if len(split_context) > 1:
#         context = split_context[0].title() + "(" + split_context[1].lower()
        
#     else:
#         context = context.title()
#     if is_name_in_db(con, context):
#         good_contexts.append(context)

# for i in tqdm.tqdm(range(len(bios_data["val"]))):
#     context = bios_data["val"][i]['input_text']['context']
#     context = context.strip().replace("-lrb- ", "(").replace(" -rrb-", ")")
#     split_context = context.split("(")
#     if len(split_context) > 1:
#         context = split_context[0].title() + "(" + split_context[1].lower()
        
#     else:
#         context = context.title()
#     if is_name_in_db(con, context):
#         good_contexts.append(context)

# np.save("../biographies/names.npy", good_contexts)

# # Be sure to close the connection
# con.close()



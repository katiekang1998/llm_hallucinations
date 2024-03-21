
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import tqdm
import urllib.parse
import pickle

pretrain_entities = np.load("/data/katie_kang/pretraining_entities/the_pile_entity_map.npz")
# names = np.load("../biographies/names.npy")
# test_idxs = np.load("../biographies/train_points_medium.npy")
# names = names[test_idxs]



with open("../biographies/train_bios.pkl", "rb") as f:
    train_bios = pickle.load(f)
names = train_bios["name"][:10000]



# titles = []
# with open("/data/katie_kang/trlx/examples/movies/titles",) as file:
#     for line in file:
#         titles.append(line.strip())

# train_idxs = np.load("/data/katie_kang/trlx/examples/movies/common_train_idxs.npy")
# test_idxs = np.load("/data/katie_kang/trlx/examples/movies/common_test_medium_idxs.npy")

# names = np.array(titles)[train_idxs][:10000]


def wikipedia_to_dbpedia(wikipedia_titles):
    dbpedia_base_url = "http://dbpedia.org/resource/"
    dbpedia_uris = []

    for title in wikipedia_titles:
        # Replace spaces with underscores and encode special characters
        encoded_title = urllib.parse.quote_plus(title.replace(" ", "_"))
        dbpedia_uri = dbpedia_base_url + encoded_title
        dbpedia_uris.append(dbpedia_uri)

    return dbpedia_uris

dbpedia_ids = wikipedia_to_dbpedia(names)


num_pretrain_entities = []
for dbpedia_id in tqdm.tqdm(dbpedia_ids):
    if dbpedia_id not in pretrain_entities:
        num_pretrain_entities.append(-1)
    else:
        num_pretrain_entities.append(len(pretrain_entities[dbpedia_id]))

np.save("../num_pretrain_entities/bios_train_points_10000.npy", num_pretrain_entities)


# num_intersections = []
# for i in tqdm.tqdm(range(len(data))):
#     result = json.loads(data[i])
#     a = []
#     for x in result['q_entities']:
#         a.append(pretrain_entities[x['URI']])
#     if len(a) >0:
#         a = np.concatenate(a)
#     b = []
#     for x in result['a_entities']:
#         b.append(pretrain_entities[x['URI']])
#     if len(b) >0:
#         b = np.concatenate(b)
#     num_intersections.append(len(list(set(a) & set(b))))
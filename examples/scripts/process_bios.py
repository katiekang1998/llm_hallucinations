import numpy as np 
import pickle

with open("../biographies/test_bios_medium_old.pkl", "rb") as f:
    test_bios_medium_old = pickle.load(f)

names = test_bios_medium_old["name"]
bios = test_bios_medium_old["bio"]

bios_new = []
for i in range(len(names)):
    name = names[i]
    bio = bios[i]
    if "(" in bio and ")" in bio:
        bio1 = len(bio.split(")")[0])+1

        bio2 = bio[bio1:]

        if "(" in name and ")" in name:
            name1 = name.split(" (")[0]
        else:
            name1 = name
        bio_new = name1 + bio2
    else:
        bio_new = bio
    bios_new.append(bio_new)


test_bios_medium = {}
test_bios_medium["name"] = names
test_bios_medium["bio"] = bios_new

with open("../biographies/test_bios_medium.pkl", "wb") as f:
    pickle.dump(test_bios_medium, f)
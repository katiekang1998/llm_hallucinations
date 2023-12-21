import openai
import re
import string
import numpy as np
from tqdm import tqdm
import time



def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def call_instructgpt_with_answers(questions, model_predictions, ground_truths):
    # if exact_match_score(model_prediction, ground_truth):
    #     return 1
    api_key = open("/data/katie_kang/openai_key_file.txt", "r").read()
    openai.api_key = api_key.strip()

    prompt_template1 = """Are the answers equivalent?

Ex 1:
Q: Who was leader of Revolutionary Army?
A1: George Washington
A2: US President Washington

Ex 2:
Q: Who is the screenwriter of Dead Silence?
A1: James Wan
A2: Leigh Whannell

Ex 3:
Q: country of citizenship of McKean?
A1: United States
A2: American

Ex 4:
Q: What genre is Taxi for Two?
A1: romantic comedy film
A2: comedy

""" 


    prompt_template2 = None

    for i in range(len(questions)):
        if i == 0:
            prompt_template2 = """Ex {}:
Q: {}
A1: {}
A2: {}

""".format(i+5, questions[i], model_predictions[i], ground_truths[i])
        else:
            prompt_template2 += """Ex {}:
Q: {}
A1: {}
A2: {}

""".format(i+5, questions[i], model_predictions[i], ground_truths[i])

    prompt_template3 = """Ex 1 Equivalent? Yes
Ex 2 Equivalent? No
Ex 3 Equivalent? Yes
Ex 4 Equivalent? Yes
"""

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

            try:
                split_response = response.split("\n")
                split_response = [split_response_i.split("? ")[1] for split_response_i in split_response]
                linguistically_equivalent = [split_response_i=="Yes" for split_response_i in split_response]
                assert(len(linguistically_equivalent) == len(questions))
            except:
                print(response)
                linguistically_equivalent = [False for _ in range(len(questions))]


            return np.array(linguistically_equivalent)
        except:
            print("EXCEPTION", num_tries)
            time.sleep(3)
            num_tries+=1



    

# data = np.load('../ckpts/sft_ctrex_llama7B_2_commit_lr1e-5_2/checkpoint_30000/hf_model/output_strings_train.npy')

data = np.load('../ckpts/sft_ctrex_llama7B_2_commit_lr1e-5_2/checkpoint_30000/hf_model/output_strings_oodsmall.npy')

questions_NEM = []
answers_NEM = []
predictions_NEM = []
exact_match_idxs = []
idk_idxs = []
not_exact_match_idxs = []
for j in tqdm(range(len(data))):
    line = data[j]
    answer = line.split('Label: ')[1].strip()
    if line.split("? ")[1].split('Label:')[0].strip() == "I don't know.":
        idk_idxs.append(j)
    else:
        try:
            prediction = line.split('The answer is')[1].split('Label:')[0].strip()[0:-1]
        except:
            print(line)
            # prediction = line.split('The Answer Is')[1].split('Label:')[0].strip()[0:-1]
            prediction = line.split('Label:')[0].strip()[0:-1]

        question = line.split('<unk>')[-1].split('?')[0].strip() + '?'

        if exact_match_score(answer, prediction):
            exact_match_idxs.append(j)
        else:
            questions_NEM.append(question)
            answers_NEM.append(answer)
            predictions_NEM.append(prediction)
            not_exact_match_idxs.append(j)

questions_NEM = np.array(questions_NEM)
answers_NEM = np.array(answers_NEM)
predictions_NEM = np.array(predictions_NEM)
not_exact_match_idxs = np.array(not_exact_match_idxs)


linguistically_equivalent_idxs = []
not_linguistically_equivalent_idxs = []

prev_equivalence = np.ones(len(data))*-1

num_points_per_prompt = 20
for i in tqdm(range(len(not_exact_match_idxs)//num_points_per_prompt, len(not_exact_match_idxs)//num_points_per_prompt+1)):
    results = call_instructgpt_with_answers(questions_NEM[i*num_points_per_prompt:(i+1)*num_points_per_prompt], 
    answers_NEM[i*num_points_per_prompt:(i+1)*num_points_per_prompt]
    , predictions_NEM[i*num_points_per_prompt:(i+1)*num_points_per_prompt])

    points_in_consideration = np.arange(i*num_points_per_prompt, min((i+1)*num_points_per_prompt, len(not_exact_match_idxs)))
    linguistically_equivalent_idxs.append(not_exact_match_idxs[points_in_consideration[np.where(results == 1)[0]]])
    not_linguistically_equivalent_idxs.append(not_exact_match_idxs[points_in_consideration[np.where(results == 0)[0]]])

    if i%20 == True:
        equivalence = prev_equivalence
        equivalence[np.array(exact_match_idxs)] = 1

        equivalence[np.concatenate(linguistically_equivalent_idxs)] = 1
        equivalence[np.concatenate(not_linguistically_equivalent_idxs)] = 0
        np.save("../ckpts/sft_ctrex_llama7B_2_commit_lr1e-5_2/checkpoint_30000/hf_model/output_strings_oodsmall_linguistic_equivalence2.npy", equivalence)
        # np.save("../ckpts/sft_ctrex_llama7B_2_commit_lr1e-5_2/checkpoint_30000/hf_model/output_strings_oodsmall_linguistic_equivalence.npy", equivalence)
        print(i, len(exact_match_idxs), len(np.concatenate(linguistically_equivalent_idxs)), len(np.concatenate(not_linguistically_equivalent_idxs)))

equivalence = prev_equivalence
equivalence[np.array(exact_match_idxs)] = 1
equivalence[np.concatenate(linguistically_equivalent_idxs)] = 1
equivalence[np.concatenate(not_linguistically_equivalent_idxs)] = 0
np.save("../ckpts/sft_ctrex_llama7B_2_commit_lr1e-5_2/checkpoint_30000/hf_model/output_strings_oodsmall_linguistic_equivalence2.npy", equivalence)
# np.save("../ckpts/sft_ctrex_llama7B_2_commit_lr1e-5_2/checkpoint_30000/hf_model/output_strings_oodsmall_linguistic_equivalence.npy", equivalence)
print(i, len(exact_match_idxs), len(np.concatenate(linguistically_equivalent_idxs)), len(np.concatenate(not_linguistically_equivalent_idxs)))

# import numpy as np
# data = np.load('../ckpts/sft_ctrex_llama7B_2_commit_lr1e-5_2/checkpoint_30000/hf_model/output_strings_train.npy')
# linguistic_equivalence = np.load("../ckpts/sft_ctrex_llama7B_2_commit_lr1e-5_2/checkpoint_30000/hf_model/output_strings_train_linguistic_equivalence.npy")

# for i in range(len(linguistic_equivalence)):
#     if linguistic_equivalence[i] == 1:
#         print(data[i])
#     elif linguistic_equivalence[i] == -1:
#         1/0
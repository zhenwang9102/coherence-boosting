import os
import re
import sys
import pickle
import logging
import tempfile
import numpy as np
import subprocess as sp

import torch

gpt2_map = {"gpt2-small": "gpt2", "gpt2-medium": "gpt2-medium", 
            "gpt2-large": "gpt2-large", "gpt2-xl": "gpt2-xl"}

gpt3_map = {"gpt3-small": "ada-msft", "gpt3-medium": "babbage-msft", 
            "gpt3-large": "curie-msft", "gpt3-xl": "davinci-msft"}

# gpt3_map = {"gpt3-small": "ada", "gpt3-medium": "babbage", 
#             "gpt3-large": "curie", "gpt3-xl": "davinci"}


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -float('inf'), logits)


def softmax(x):
    # x: [D, ]
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def logprob_contrast(l1, l2, alpha):
    return l1 - alpha * l2


def batching(data, bs):
    for i in range(0, len(data), bs):
        yield data[i: i+bs]


def construct_fs_prompt(task, cur_data):
    # create few shot prompts with description
    description = (task.get_description() + "\n=====\n") if task.add_description and task.get_description() else ""

    if task.num_shots == 0:
        fewshots = ""
    else:
        if task.has_train_set:
            fewshots = [task.standardize(x) for x in task.rnd.sample(list(task.get_train_set()), task.num_shots)]
        else:
            sample_pool = task.get_val_set() if task.has_val_set else task.get_test_set()
            samples = [task.standardize(x) for x in task.rnd.sample(list(sample_pool), task.num_shots + 1)]
            fewshots = [x for x in samples if x['text'] != cur_data['text']][:task.num_shots]

        fewshots = "\n\n".join([data['context'] + " " +  data['target'] for data in fewshots]) + "\n\n"

    # cur_data['input'] = "<|endoftext|> " + description + fewshots + cur_data['context']
    if not task.add_description and task.num_shots == 0:
        cur_data['input']  = cur_data['context'] 
    else:
        cur_data['input'] = description + fewshots + cur_data['context']
    # test -> context -> input
    return cur_data


def get_cache_file(task_name, model_name, n_sample, catch_prefix, data_prefix, **kwargs):
    prefix = os.path.join(catch_prefix, task_name)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    return os.path.join(prefix, "{}_{}_{}_{}.pkl".format(task_name, 
                                                         model_name,
                                                         data_prefix,
                                                         n_sample))


def get_pseudo_val_file(task_name, n_sample):
    prefix = os.path.join('./data', task_name)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    return os.path.join(prefix, "{}_pval_{}.pkl".format(task_name, n_sample))


def detokenizer(string):
    # ari custom
    string = string.replace("`` ", '"')
    string = string.replace(" ''", '"')
    string = string.replace("` ", '"')
    string = string.replace(" ' ", '" ')
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" :", ":")
    string = string.replace(" ;", ";")
    string = string.replace(" .", ".")
    string = string.replace(" !", "!")
    string = string.replace(" ?", "?")
    string = string.replace(" ,", ",")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    # string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    # string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    # ari custom
    string = string.replace(" n't ", "n't ")
    string = string.replace(" 'd ", "'d ")
    string = string.replace(" 'm ", "'m ")
    string = string.replace(" 're ", "'re ")
    string = string.replace(" 've ", "'ve ")
    return string
    
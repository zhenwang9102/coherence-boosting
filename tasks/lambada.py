import os
import sys
import json
import random
import collections
import numpy as np
from best_download import download_file
from pytablewriter import MarkdownTableWriter, LatexTableWriter

import torch
import datasets
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

from utils import *
from task import LAMBADATask


class LAMBADA(LAMBADATask):
    def __init__(self, add_description, num_shots, rnd, **kwargs):
        super().__init__()
        self.task_name = "LAMBADA"
        self.add_description = add_description
        self.num_shots = num_shots
        self.rnd = rnd

        self.has_train_set = True
        self.has_val_set = True
        self.has_test_set = True

        self.load_data()

    def load_data(self):
        os.system("mkdir -p data/lambada")
        try:
            if not os.path.exists("data/lambada/lambada_test.jsonl"):
                download_file(
                    "http://eaidata.bmk.sh/data/lambada_test.jsonl", 
                    "data/lambada/lambada_test.jsonl", 
                    "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226"
                )
        except:
            # fallback - for some reason best_download doesnt work all the time here
            os.system("wget http://eaidata.bmk.sh/data/lambada_test.jsonl -O data/lambada/lambada_test.jsonl")
            os.system('echo "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226  data/lambada/lambada_test.jsonl" | sha256sum --check')
        
        self.data = datasets.load_dataset('lambada')

    def preprocess(self, text):
        text = text.replace("“", '"')
        text = text.replace("”", '"')
        text = text.replace("''", '"')
        text = text.replace("``", '"')
        return '\n'+text.strip()

    def get_train_set(self):
        return self.data['train']

    def get_val_set(self):
        return self.data['validation']

    def get_test_set(self):
        data = []
        with open("data/lambada/lambada_test.jsonl") as fh:
            for line in fh:
                data.append(self.preprocess(json.loads(line)['text'].strip()))
        return data

    def task_description(self):
        return ""

    def standardize(self, sample):
        # val: {'text'}
        # test: ''
        cur_data = {}
        if isinstance(sample, str):
            input_ids = tokenizer.encode(sample)
        else:
            input_ids = tokenizer.encode(sample['text'])

        cur_data['context'] = input_ids[:-1]
        cur_data['target'] =  [input_ids[-1]]
        
        return cur_data

    def get_contrast_ctx(self, sample, short_len=12):
        return sample['context'][-short_len::]
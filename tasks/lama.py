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

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from utils import *
from task import LAMBADATask


class LAMA(LAMBADATask):
    def __init__(self, add_description, num_shots, rnd, **kwargs):
        super().__init__()
        self.task_name = "LAMA"
        self.add_description = add_description
        self.num_shots = num_shots
        self.rnd = rnd

        self.has_train_set = True
        self.has_val_set = True
        self.has_test_set = True

        relations = {'P19': '[X] was born in [Y] .', 
                          'P20': '[X] died in [Y] .', 
                          'P279': '[X] is a subclass of [Y] .', 
                          'P37': 'The official language of [X] is [Y] .', 
                          'P413': '[X] plays in [Y] position .', 
                          #  'P166': '[X] was awarded the [Y] .',  # not stored
                          'P449': '[X] was originally aired on [Y] .', 
                          #  'P69': '[X] was educated at the University of [Y] .',  # not stored
                          'P47': '[X] shares border with [Y] .', 
                          'P138': '[X] is named after [Y] .', 
                          'P364': 'The original language of [X] is [Y] .', 
                          #  'P54': '[X] plays with [Y] .',  # not stored
                          'P463': '[X] is a member of [Y] .', 
                          'P101': '[X] works in the field of [Y] .', 
                          #  'P1923': '[Y] participated in the [X] .',  # not stored
                        #    'P106': '[X] is a [Y] by profession .',  # revise 
                          'P106': 'The profession of [X] is [Y] .', 
                          'P527': '[X] consists of [Y] .', 
                          #  'P102': '[X] is a member of the [Y] political party .',  # not stored
                          'P530': '[X] maintains diplomatic relations with [Y] .', 
                          'P176': '[X] is produced by [Y] .', 
                        #    'P27': '[X] is [Y] citizen .',   # revise
                          'P27': '[X] is a citizen of [Y] .', 
                          'P407': '[X] was written in [Y] .', 
                          'P30': '[X] is located in [Y] .', 
                          'P178': '[X] is developed by [Y] .', 
                          'P1376': '[X] is the capital of [Y] .', 
                          'P131': '[X] is located in [Y] .', 
                          'P1412': '[X] used to communicate in [Y] .', 
                          'P108': '[X] works for [Y] .', 
                          'P136': '[X] plays [Y] music .', 
                          'P17': '[X] is located in [Y] .', 
                          'P39': '[X] has the position of [Y] .', 
                          'P264': '[X] is represented by music label [Y] .', 
                          'P276': '[X] is located in [Y] .', 
                          'P937': '[X] used to work in [Y] .', 
                          'P140': '[X] is affiliated with the [Y] religion .', 
                          'P1303': '[X] plays [Y] .', 
                          'P127': '[X] is owned by [Y] .', 
                          'P103': 'The native language of [X] is [Y] .', 
                          #  'P190': '[X] and [Y] are twin cities .',  # bad template
                          'P1001': '[X] is a legal term in [Y] .', 
                          'P31': '[X] is a [Y] .', 
                          'P495': '[X] was created in [Y] .', 
                          'P159': 'The headquarter of [X] is in [Y] .', 
                          'P36': 'The capital of [X] is [Y] .', 
                          'P740': '[X] was founded in [Y] .', 
                          'P361': '[X] is part of [Y] .'}

        self.relations = relations
        self.max_len = max(kwargs['slen_list'])

        self.load_data()

    def prepare_prompt(self, x, template):
        x_pos = template.find('[X]')
        y_pos = template.find('[Y]')
        seg1 = template[0: x_pos]
        seg2 = template[x_pos+3: y_pos]
        return f"{seg1}{x}{seg2}"[:-1]

    def load_data(self):
        prefix = './data/lama/original_rob'
        rela_ids = self.relations.keys()
        train, val, test = [], [], []
        for i, rela in enumerate(rela_ids):
            rela_path = os.path.join(prefix, rela)
            with open(rela_path + '/train.jsonl') as f:
                for line in f:
                    res = json.loads(line)
                    prompt = self.prepare_prompt(res['sub_label'], self.relations[rela])
                    if len(tokenizer.encode(" " + res['obj_label'])) != 1 or len(tokenizer.encode(prompt)) <= self.max_len:
                        continue
                    train.append({'context': prompt, 'target': res['obj_label']})

            with open(rela_path + '/dev.jsonl') as f:
                for line in f:
                    res = json.loads(line)
                    prompt = self.prepare_prompt(res['sub_label'], self.relations[rela])
                    if len(tokenizer.encode(" " + res['obj_label'])) != 1 or len(tokenizer.encode(prompt)) <= self.max_len:
                        continue
                    val.append({'context': prompt, 'target': res['obj_label']})

            with open(rela_path + '/test.jsonl') as f:
                for line in f:
                    res = json.loads(line)
                    prompt = self.prepare_prompt(res['sub_label'], self.relations[rela])
                    if len(tokenizer.encode(" " + res['obj_label'])) != 1 or len(tokenizer.encode(prompt)) <= self.max_len:
                        continue
                    test.append({'context': prompt, 'target': res['obj_label']})

        print(len(train), len(val), len(test))
        self.data = {'train': train, 'validation': val, 'test': test}

    def get_train_set(self):
        return self.data['train']

    def get_val_set(self):
        return self.data['validation']

    def get_test_set(self):
        return self.data['test']

    def task_description(self):
        return ""

    def standardize(self, sample):
        cur_data = {}
        cur_data['original_context'] = sample['context'] + " " + sample['target']  # ensure the space before last token
        input_ids = tokenizer.encode(cur_data['original_context'])
        cur_data['context'] = input_ids[:-1]
        cur_data['target'] =  [input_ids[-1]]
        
        return cur_data

    def get_contrast_ctx(self, sample, short_len=12):
        return sample['context'][-short_len::]
import os
import re
import sys
import json
import random
import numpy as np
from best_download import download_file
from pytablewriter import MarkdownTableWriter, LatexTableWriter

import torch
import datasets
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

from utils import *
from task import MultipleChoiceTask


class TREC(MultipleChoiceTask):
    def __init__(self, add_description, num_shots, rnd, **kwargs):
        super().__init__()
        self.task_name = "TREC"
        self.add_description = add_description
        self.num_shots = num_shots
        self.rnd = rnd

        self.has_train_set = True
        self.has_val_set = False
        self.has_test_set = True

        self.load_data()

    def load_data(self):
        self.data = {'train': [], 'validation': [], 'test': []}

        train_file = './data/trec/train.txt'
        if os.path.exists(train_file):
            with open(train_file) as f:
                for line in f:
                    label = line[:line.index(' ')].split(':')[0]
                    question = detokenizer(line[line.index(' ') + 1:]).strip()
                    self.data['train'].append({'label': label, 
                                               'question': question})
        else:
            raise FileNotFoundError("Train data not found!")

        test_file = './data/trec/test.txt'
        if os.path.exists(test_file):
            with open(test_file) as f:
                for line in f:
                    label = line[:line.index(' ')].split(':')[0]
                    question = detokenizer(line[line.index(' ') + 1:]).strip()
                    self.data['test'].append({'label': label, 
                                              'question': question})
        else:
            raise FileNotFoundError("Testing data not found!")


    def get_train_set(self):
        return self.data['train']

    def get_val_set(self):
        return self.data['validation']

    def get_test_set(self):
        return self.data['test']

    def get_description(self):
        return ""

    def standardize(self, sample):

        label2desc = {'DESC': 'a description.', 'ENTY': 'an entity.', 'LOC': 'a location.', 'NUM': 'a number.', 'ABBR': 'an abbreviation.', 'HUM': 'a person.'}
        label2idx = {'DESC': 0, 'ENTY': 1, 'LOC': 2, 'NUM': 3, 'ABBR': 4, 'HUM': 5}

        cur_data = {
            "text": sample['question'],
            "choices": list(label2desc.values()),
            "label": label2idx[sample['label']]
        }

        cur_data['context'] = "{} The answer to this question will be".format(cur_data['text'])
        cur_data['target'] = cur_data['choices'][cur_data['label']]

        return cur_data

    def get_contrast_ctx(self, batch):
        n_sents = sum([len(x['choices']) for x in batch])
        prompts = [' The answer to this question will be'] * n_sents 
        return prompts

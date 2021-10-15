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


class SST2(MultipleChoiceTask):
    def __init__(self, add_description, num_shots, rnd, **kwargs):
        super().__init__()
        self.task_name = "SST-2"
        self.add_description = add_description
        self.num_shots = num_shots
        self.rnd = rnd

        self.has_train_set = True
        self.has_val_set = True
        self.has_test_set = True
        self.answer_set = ['negative\n', 'positive\n']

        self.load_data()

    def load_data(self):
        self.data = datasets.load_dataset('sst', 'default')

    def get_train_set(self):
        return self.filter(self.data['train'])

    def get_val_set(self):
        return self.filter(self.data['validation'])

    def get_test_set(self):
        return self.filter(self.data['test'])

    def get_description(self):
        return ""

    def filter(self, data):
        return [x for x in data if self.labeling(x['label']) != -1]

    def labeling(self, score):
        if 0 <= score <=0.4:
            return 0
        elif 0.4 < score <= 0.6: # neutral class
            return -1
        elif 0.6 < score <= 1: 
            return 1
        else: 
            raise Exception('wrong score!')

    def standardize(self, sample):
        cur_data = {
            "text": sample['sentence'],
            "choices": self.answer_set,
            "label": self.labeling(sample['label'])
        }

        cur_data['context'] = "{} This quote has a tone that is".format(cur_data['text'])
        cur_data['target'] = cur_data['choices'][cur_data['label']]

        return cur_data

    def get_contrast_ctx(self, batch):
        n_sents = sum([len(x['choices']) for x in batch])
        prompts = [' This quote has a tone that is'] * n_sents 
        return prompts


class SST5(SST2):
    def __init__(self, add_description, num_shots, rnd, **kwargs):
        self.task_name = "SST-5"
        self.add_description = add_description
        self.num_shots = num_shots
        self.rnd = rnd

        self.has_train_set = True
        self.has_val_set = True
        self.has_test_set = True
        self.answer_set = ['very negative.<|endoftext|>', 
                           'somewhat negative.<|endoftext|>', 
                           'neutral.<|endoftext|>', 
                           'somewhat positive.<|endoftext|>', 
                           'very positive.<|endoftext|>']
        self.load_data()

    def filter(self, data):
        return data

    def labeling(self, score):
        if 0 <= score <=0.2: 
            return 0
        elif 0.2 < score <=0.4: 
            return 1
        elif 0.4 < score <= 0.6: 
            return 2
        elif 0.6 < score <= 0.8: 
            return 3
        elif 0.8 < score <=1: 
            return 4
        else: 
            raise Exception('wrong score!')

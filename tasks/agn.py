import os
import re
import sys
import json
import random
import numpy as np
import pandas as pd
from best_download import download_file
from pytablewriter import MarkdownTableWriter, LatexTableWriter

import torch
import datasets
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

from utils import *
from task import MultipleChoiceTask


class AGNews(MultipleChoiceTask):
    def __init__(self, add_description, num_shots, rnd, **kwargs):
        super().__init__()
        self.task_name = "AGNews"
        self.add_description = add_description
        self.num_shots = num_shots
        self.rnd = rnd

        self.has_train_set = True
        self.has_val_set = False
        self.has_test_set = True

        self.load_data()

    def load_data(self):
        self.data = {'train': [], 'validation': [], 'test': []}

        train_file = './data/agn/train.csv'
        if os.path.exists(train_file):
            data = pd.read_csv(train_file)
            for idx, row in data.iterrows():
                self.data['train'].append(dict(row))
        else:
            raise FileNotFoundError("Train data not found! Please put the official train file to ./data/agn")

        test_file = './data/agn/test.csv'
        if os.path.exists(test_file):
            data = pd.read_csv(test_file)
            for idx, row in data.iterrows():
                self.data['test'].append(dict(row))
        else:
            raise FileNotFoundError("Testing data not found! Please put the official test file to ./data/agn")

    def get_train_set(self):
        return self.data['train']

    def get_val_set(self):
        return self.data['validation']

    def get_test_set(self):
        return self.data['test']

    def get_description(self):
        return ""

    def standardize(self, sample):

        cur_data = {
            "text": sample['Description'],
            "choices": ['World', 'Sports', 'Business', 'Science' ] ,
            "label": int(sample['Class Index']) - 1
        }

        cur_data['context'] = "Title: {}\nSummary: {}\nTopic:".format(sample['Title'], sample['Description'])
        cur_data['target'] = cur_data['choices'][cur_data['label']]

        return cur_data

    def get_contrast_ctx(self, batch):
        n_sents = sum([len(x['choices']) for x in batch])
        prompts = [' Topic:'] * n_sents 
        return prompts

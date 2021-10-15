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


class StoryCloze(MultipleChoiceTask):
    def __init__(self, add_description, num_shots, rnd, **kwargs):
        super().__init__()
        self.task_name = 'StoryCloze'
        self.add_description = add_description
        self.num_shots = num_shots
        self.rnd = rnd

        self.has_train_set = False
        self.has_val_set = True
        self.has_test_set = True

        self.load_data()

    def load_data(self):
        self.data = {'train': [], 'validation': [], 'test': []}

        # val_file = './data/storycloze/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv'
        val_file = './data/storycloze/cloze_test_val__spring2016 - cloze_test_ALL_val.csv'
        if os.path.exists(val_file):
            data = pd.read_csv(val_file)
            for idx, row in data.iterrows():
                self.data['validation'].append(dict(row))
        else:
            raise FileNotFoundError("Validation data not found!")

        test_file = './data/storycloze/cloze_test_test__spring2016 - cloze_test_ALL_test.csv'
        if os.path.exists(test_file):
            data = pd.read_csv(test_file)
            for idx, row in data.iterrows():
                self.data['test'].append(dict(row))
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
        cur_data = {
            "text": "{} {} {} {}".format(sample['InputSentence1'], sample['InputSentence2'], 
                                         sample['InputSentence3'], sample['InputSentence4']),
            "choices": [sample["RandomFifthSentenceQuiz1"], sample["RandomFifthSentenceQuiz2"]],
            "label": int(sample['AnswerRightEnding']) - 1
        }

        cur_data['context'] = cur_data["text"]
        cur_data['target'] = cur_data['choices'][cur_data['label']]

        return cur_data

    def get_contrast_ctx(self, batch):
        n_sents = len(batch) * len(batch[0]['choices'])
        prompts = ['<|endoftext|>'] * n_sents 
        return prompts

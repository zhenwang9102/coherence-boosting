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


class HellaSwag(MultipleChoiceTask):
    def __init__(self, add_description, num_shots, rnd, **kwargs):
        super().__init__()
        self.task_name = "HellaSwag"
        self.add_description = add_description
        self.num_shots = num_shots
        self.rnd = rnd

        self.has_train_set = True
        self.has_val_set = True
        self.has_test_set = False

        self.load_data()

    def load_data(self):
        self.data = datasets.load_dataset('hellaswag')

    def get_train_set(self):
        return self.data['train']

    def get_val_set(self):
        return self.data['validation']

    def get_test_set(self):
        return self.data['test']

    def preprocess(self, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        # text = text.replace(" [title]", ".")
        text = re.sub('\\[.*?\\]', '', text)
        text = text.replace("  ", " ")
        return text

    def get_description(self):
        return ""

    def standardize(self, sample):
        ctx = sample["ctx_a"] + " " + sample["ctx_b"].capitalize()
        cur_data = {
            "text": self.preprocess(sample['activity_label'] + ': ' + ctx),
            "choices": [self.preprocess(ending) for ending in sample['endings']],
            "label": int(sample['label']),
        }

        cur_data['context'] = cur_data['text'] 
        cur_data['target'] =  cur_data['choices'][cur_data['label']]

        return cur_data

    def get_contrast_ctx(self, batch):
        n_sents = len(batch) * len(batch[0]['choices'])
        prompts = []
        for data in batch:
            temp = data['input'][data['input'].rfind('.')+1:]
            if temp:
                prompts += [temp] * len(data['choices'])
            else:
                prompts += ['<|endoftext|>'] * len(data['choices'])
        return prompts

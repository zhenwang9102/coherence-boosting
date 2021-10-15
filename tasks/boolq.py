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


class BoolQ(MultipleChoiceTask):
    def __init__(self, add_description, num_shots, rnd, **kwargs):
        super().__init__()
        self.task_name = "BoolQ"
        self.add_description = add_description
        self.num_shots = num_shots
        self.rnd = rnd

        self.has_train_set = True
        self.has_val_set = True
        self.has_test_set = False

        self.load_data()

    def load_data(self):
        self.data = datasets.load_dataset('super_glue', 'boolq')

    def get_train_set(self):
        return self.data['train']

    def get_val_set(self):
        return self.data['validation']

    def get_test_set(self):
        return self.data['test']

    def get_description(self):
        return ""

    def proc_question(self, s):
        s = s[0].upper() + s[1:]
        s = s.replace(' i ', ' I ')
        s = s + '?'
        return s

    def standardize(self, sample):
        question = self.proc_question(sample["question"])
        cur_data = {
            "question": question,
            "text": question,
            "choices": ["no", "yes"],
            "label": sample['label']
        }
        
        cur_data['context'] = "Title: {}\n Question: {}?\n The answer is:".format(sample['passage'], question) 
        cur_data['target'] = cur_data['choices'][cur_data['label']]

        return cur_data

    def get_contrast_ctx(self, batch):
        # n_sents = len(batch) * len(batch[0]['choices'])
        # prompts = ['<|endoftext|>'] * n_sents
        prompts = []
        for data in batch:
            choices = [" Question: {}?\n The answer is:".format(data['question'])] * len(data['choices'])
            prompts += choices

        return prompts

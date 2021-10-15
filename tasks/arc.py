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


class ARCEasy(MultipleChoiceTask):
    def __init__(self, add_description, num_shots, rnd, **kwargs):
        super().__init__()
        self.task_name = "ARC Easy"
        self.add_description = add_description
        self.num_shots = num_shots
        self.rnd = rnd

        self.has_train_set = True
        self.has_val_set = True
        self.has_test_set = True

        self.load_data()

    def load_data(self):
        self.data = datasets.load_dataset('ai2_arc', 'ARC-Easy')

    def get_train_set(self):
        return self.data['train']

    def get_val_set(self):
        return self.data['validation']

    def get_test_set(self):
        return self.data['test']

    def get_description(self):
        return ""

    def standardize(self, sample):
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        sample["answerKey"] = num_to_letter.get(sample["answerKey"], sample["answerKey"])
        cur_data = {
            "text": sample['question'],
            "choices": sample["choices"]["text"],
            "label": ["A", "B", "C", "D", "E"].index(sample["answerKey"])
        }

        cur_data['context'] = "Question: " + cur_data["text"] + "\nAnswer:" 
        cur_data['target'] = cur_data['choices'][cur_data['label']]

        return cur_data

    def get_contrast_ctx(self, batch):
        n_sents = sum([len(x['choices']) for x in batch])
        prompts = [' Answer:'] * n_sents 
        return prompts


class ARCChallenge(ARCEasy):
    def __init__(self, add_description, num_shots, rnd, **kwargs):
        self.task_name = "ARC Challenge"
        self.add_description = add_description
        self.num_shots = num_shots
        self.rnd = rnd

        self.has_train_set = True
        self.has_val_set = True
        self.has_test_set = True

        self.load_data()

    def load_data(self):
        self.data = datasets.load_dataset('ai2_arc', 'ARC-Challenge')

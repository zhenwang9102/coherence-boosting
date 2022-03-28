import os
import sys
import argparse
import numpy as np

from utils import *


def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]


def config():
    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--tasks', type=str, default='', help='A list of tasks, seperated by the semicolon')
    parser.add_argument('--models', type=str, default='', help='A list of models, seperated by the semicolon')
    parser.add_argument('--model_name', type=str, default='', help='Currently support: gpt2-small, gpt2-medium, gpt2-large, gpt2-xl, gpt3-small, gpt3-medium, gpt3-large, gpt3-xl')
    parser.add_argument('--task', type=str, default='', help='Currently support: lambada, storycloze, csqa, rte, cb, sst2, sst5, trec, agn, boolq, copa, piqa, arc_easy, arc_challenge, hellaswag, openbookqa, lama')
    
    parser.add_argument("--alpha_start", type=float, default=-2)
    parser.add_argument("--alpha_end", type=float, default=3)
    parser.add_argument("--alpha_step", type=float, default=0.1)

    parser.add_argument("--slen_start", type=int, default=1)
    parser.add_argument("--slen_end", type=int, default=20)
    parser.add_argument("--slen_step", type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=2, help='Batch size depending on the memory and model size')
    parser.add_argument('--len_norm', type='bool', default=False, help='Whether to apply the length normalization')
    parser.add_argument('--add_description', type='bool', default=False, help='Task instruction to be added in the head of the prompt')
    parser.add_argument('--num_shots', type=int, default=0, help='Number of shots for few-shot prompting')
    parser.add_argument('--split', type=str, default='test', help='train, validation, test')
    parser.add_argument('--key_file', type=str, default='./api_key.txt', help='File to store the GPT-3 API Key')

    parser.add_argument('--use_val', type='bool', default=True, help='Whether to use the validation set')
    parser.add_argument('--printing', type='bool', default=False, help='Whether to print results for all alphas')
    parser.add_argument('--verbose', type='bool', default=True, help='Whether to show the progress bar')
    parser.add_argument('--demo', type=int, default=0, help='Number of samples to debug the code quickly; set it to 0 to evaluate full dataset')
    parser.add_argument('--peek', type='bool', default=False, help='Debugging the prompt format by printing one example')
    parser.add_argument('--cache_file', type=str, default='./cache', help='File path to store the cached results')
    parser.add_argument('--use_cache', type='bool', default=False, help='Whether to use stored cached results')
    parser.add_argument('--val_size', type=int, default=0, help='Number of validation samples; set it to 0 to use the same amount of testing samples')

    args = parser.parse_args()

    print('Raw Arguments: ', args)
    print('Process ID: ', os.getpid())

    args = vars(args)

    args['tasks'] = [x for x in args['tasks'].strip().split(';') if x != '']

    if args['models'] == 'all':
        args['models'] = list(gpt2_map.keys()) + list(gpt3_map.keys())
    elif args['models'] == 'gpt2':
        args['models'] = list(gpt2_map.keys())
    elif args['models'] == 'gpt3':
        args['models'] = list(list(gpt3_map.keys()))
    else:
        args['models'] = args['models'].strip().split(';')

    if 'gpt3' in [x[:4] for x in args['models']]:
        if not os.path.exists(args['key_file']):
            raise FileNotFoundError(f"Please store your API key in {args['key_file']}")
        args['gpt3_apikey'] = open(args['key_file']).readlines()[0].strip()

    # quick args preprocessing
    args['alpha_list'] = [round(x, 2) for x in np.arange(args['alpha_start'], args['alpha_end'] + args['alpha_step'], args['alpha_step'])]
    args['slen_list'] = list(range(args['slen_start'], args['slen_end'] + args['slen_step'], args['slen_step']))

    return args
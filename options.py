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
    parser.add_argument('--model_name', type=str, default='gpt2', help='gpt2, gpt2-medium, gpt2-large, gpt2-xl, \
                                                                        gpt3-small, gpt3-medium, gpt3-large, gpt3-xl')
    parser.add_argument('--task', type=str, default='lambada', help='')
    parser.add_argument('-ad', '--add_description', type='bool', default=False, help='')
    parser.add_argument('-ns', '--num_shots', type=int, default=0, help='0, 1, 4, 8')
    parser.add_argument('--split', type=str, default='test', help='train, validation, test')

    parser.add_argument("--alpha_start", type=float, default=-2)
    parser.add_argument("--alpha_end", type=float, default=3)
    parser.add_argument("--alpha_step", type=float, default=0.1)

    parser.add_argument("--slen_start", type=int, default=1)
    parser.add_argument("--slen_end", type=int, default=20)
    parser.add_argument("--slen_step", type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2, help='')

    parser.add_argument('--key_file', type=str, default='./api_key.txt', help='')
    parser.add_argument('--demo', type=int, default=0, help='set it it 0 to evaluate full dataset')
    parser.add_argument('--printing', type='bool', default=False, help='')
    parser.add_argument('--verbose', type='bool', default=True)
    parser.add_argument('--peek', type='bool', default=False)
    parser.add_argument('--len_norm', type='bool', default=False)
    parser.add_argument('--cache_file', type=str, default='./cache')
    parser.add_argument('--use_cache', type='bool', default=False)

    parser.add_argument('--tasks', type=str, default='', help='a list of tasks')
    parser.add_argument('--models', type=str, default='', help='a list of models')
    parser.add_argument('--mode', type=int, default=0, help='0: ours, 1: alpha=0, 2: alpha=1, 3: alpha=0, LenNorm')
    parser.add_argument('--val_size', type=int, default=0)
    parser.add_argument('--use_val', type='bool', default=True)

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
        args['gpt3_apikey'] = open(args['key_file']).readlines()[0].strip()        

    # quick args preprocessing
    args['alpha_list'] = [round(x, 2) for x in np.arange(args['alpha_start'], args['alpha_end'] + args['alpha_step'], args['alpha_step'])]
    args['slen_list'] = list(range(args['slen_start'], args['slen_end'] + args['slen_step'], args['slen_step']))

    return args
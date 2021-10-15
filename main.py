import os
import sys
import json
import random
import pickle
import argparse
import collections
import numpy as np
from tqdm import tqdm
from pytablewriter import TsvTableWriter

from evaluator import evaluator
from options import config
from utils import *
import tasks


def main(args):
    args['rnd'] = random.Random()
    args['rnd'].seed(args['random_seed'])

    task_list = args['tasks']
    model_list = args['models']

    res = collections.defaultdict(dict)

    for task_name in task_list:
        args['task'] = task_name
        task = tasks.get_task(args['task'])(**args)

        for model_name in model_list:
            args['model_name'] = model_name

            if model_name[3] == '3':
                args['batch_size'] = 32
            else:
                args['batch_size'] = 2
            
            # prepare testing and validation sets
            if task.has_test_set and task.has_val_set:
                val_data = task.get_val_set()
                test_data = task.get_test_set()
                test_prefix, val_prefix = 'test', 'val'
            else:
                if task.has_test_set:
                    test_data = task.get_test_set()
                    test_prefix = 'test'
                else:
                    test_data = task.get_val_set()
                    test_prefix = 'ptest'
                
                val_prefix = 'pval'
                if args['val_size']:
                    val_size = max(args['val_size'], len(list(test_data)))
                else:
                    val_size = len(list(test_data))
                temp_val_file = get_pseudo_val_file(args['task'], val_size)
                assert task.has_train_set
                
                if os.path.exists(temp_val_file):
                    val_data = pickle.load(open(temp_val_file, 'rb'))
                else:
                    train_data = task.get_train_set()
                    val_data = random.sample(list(train_data), val_size)
                    pickle.dump(val_data, open(temp_val_file, 'wb'),protocol=-1)

            set_params = {}
            if args['use_val']:
                print('Using **{}** set for Task **{}**.'.format(('Validation' if val_prefix != 'pval' else 'Pseudo Validation'), task.task_name))
                val_res = evaluator(args, task, val_data, data_prefix=val_prefix)
                if 'alpha' in val_res:
                    set_params['alpha'] = val_res['alpha']
                if 'short_len' in val_res:       
                    set_params['short_len'] = val_res['short_len']
                
            print('Using **{}** set for Task **{}**.'.format(('Testing' if test_prefix!= 'ptest' else 'Pseudo Testing'), task.task_name))
            test_res = evaluator(args, task, test_data, data_prefix=test_prefix, set_params=set_params)

            res[task_name][model_name] = test_res['acc']

    # nice printing
    csv_writer = TsvTableWriter()
    table_header = 'Best Alpha'
    csv_writer.headers = [table_header] + model_list
    values = []
    for task in res:
        values.append([task] + [res[task][x] for x in model_list])
    csv_writer.value_matrix = values
    print(csv_writer.dumps())


    return

if __name__ == '__main__':
    main(config())

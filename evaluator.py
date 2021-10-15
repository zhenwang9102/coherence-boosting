import os
import sys
import json
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pygments import console

from options import config
from utils import *
import tasks
import models


def evaluator(args, task, eval_data, data_prefix='', set_params={}):

    # restore cached results if exists
    cache_file = get_cache_file(args['task'], args['model_name'], len(eval_data), args['cache_file'], data_prefix)
    if os.path.exists(cache_file) and not args['demo']:
        print('Reusing results from file: {}'.format(cache_file))
        res = pickle.load(open(cache_file, 'rb'))
        metrics = task.aggregating(res, **args)

        return task.return_results(metrics, set_params, **args)
    else:  # skip unstored restuls
        print(console.colorize("green", "Cache file not found: {}".format(cache_file)))
        if args['use_cache']: # force to use the cache
            return -1, None
                    
    if args['demo']:
        eval_data = list(eval_data)[:args['demo']]


    final_data = []
    for id, sample in tqdm(enumerate(eval_data), disable=True):
        cur_data = task.standardize(sample)
        cur_data = construct_fs_prompt(task, cur_data)
        cur_data['id'] = id
        final_data.append(cur_data)

    if args['peek']:
        cur_data = random.choice(final_data)
        print(cur_data)
        exit()
    
    # load the model
    lm = models.get_model(args['model_name'])(**args)

    # inference
    batched_data = list(batching(final_data, args['batch_size']))
    res = []
    for i, batch in tqdm(enumerate(batched_data), total=len(batched_data), disable=not args['verbose']):
        cur_res = task.contrasting(lm, batch, **args)
        res.append(cur_res)

    # cache results
    if not args['demo']:
        pickle.dump(res, open(cache_file, 'wb'), protocol=-1)
        
    # calculate metircs
    metrics = task.aggregating(res, **args)

    return task.return_results(metrics, set_params, **args)

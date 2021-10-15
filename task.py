import os
import sys
import numpy as np
from pytablewriter import TsvTableWriter

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

from utils import *

class MultipleChoiceTask():

    def contrasting(self, lm, batch, **kwargs):
        n_batch = len(batch)
        contexts, choices, n_choices = [], [], [0]
        for data in batch:
            n_choices.append(len(data['choices']))
            contexts += [data['input']] * len(data['choices'])
            choices += data['choices']
        choices = [" " + x for x in choices]
        n_choices = np.cumsum(n_choices)

        assert len(contexts) == len(choices), "{} != {}".format(len(contexts), len(choices))
        full_logprobs = lm.loglikelihood(contexts, choices)  # list of [1, L', V]

        # Provide different contrastive contexts
        contra_ctx = self.get_contrast_ctx(batch)
        contra_logprobs = lm.loglikelihood(contra_ctx, choices)  # list of [1, L', V]

        contrast_logprobs = []  # [d1 ->[c1->[long, short],c2->[long, short],c2->[long, short]], d2 -> []]
        labels = []
        choice_lens = []
        for i in range(n_batch):
            preds = []
            lens = []
            for k in range(n_choices[i], n_choices[i+1]):
                full, contra, target = full_logprobs[k], contra_logprobs[k], choices[k]
                lp1, lp2 = lm.continuation_ll(full, target, contra)
                preds.append([lp1, lp2])
                lens.append(len(target))

            contrast_logprobs.append(preds)
            labels.append(batch[i]['label'])
            choice_lens.append(lens)

        return (contrast_logprobs, labels, choice_lens)

    def aggregating(self, results, alpha_list, len_norm, **kwargs):
        # aggregate results
        acc = np.zeros(len(alpha_list))
        data = [y for x in results for y in x[0]]
        labels = [y for x in results for y in x[1]]
        choice_lens = [y for x in results for y in x[2]]

        res = []
        for i, (lp, label, lens) in enumerate(zip(data, labels, choice_lens)):
            for j, alpha in enumerate(alpha_list):
                preds = map(logprob_contrast, [x[0] for x in lp], [x[1] for x in lp], [alpha] * len(lp))
                if len_norm:
                    acc[j] += np.argmax(list(preds) / np.array(lens)) == label
                else:
                    acc[j] += np.argmax(list(preds)) == label

        acc /= len(data)

        return acc

    def return_results(self, results, set_params={}, printing=True, alpha_list=[], **kwargs):
        acc = results
        alpha_dict = {i: v for i, v in enumerate(alpha_list)}
        acc_best = np.argmax(acc, axis=0)

        if printing:

            print('Task: {}, Model: {}, Description: {}, Few-shot: {}'.format(
                kwargs['task'], kwargs['model_name'], kwargs['add_description'], kwargs['num_shots']
            ))
            print('Alpha\tAccuracy')
            for idx, _alpha in alpha_dict.items():
                print("{}\t{:.5f}".format(_alpha, acc[idx]))

        if 'alpha' in set_params:
            alpha = set_params['alpha']
            print("Acc={} at alpha={}".format(acc[alpha_list.index(alpha)], alpha))
            return {'acc': acc[alpha_list.index(alpha)], 'alpha': alpha, 'acc_list': acc}
        else:
            print("Best acc={} at alpha={}".format(acc[acc_best], alpha_dict[acc_best]))
            return {'acc': acc[acc_best], 'alpha': alpha_dict[acc_best], 'acc_list': acc}



class LAMBADATask():

    def contrasting(self, lm, batch, slen_list=[], alpha_list=[], **kwargs):
        # logprobs for full contexts
        full_contexts = [x['input'] for x in batch]
        targets = [x['target'] for x in batch]
        full_logprobs = lm.loglikelihood(full_contexts, targets, sf_norm=False)  # list of [1, 1, V]

        ppl = np.zeros((len(slen_list), len(alpha_list)))
        acc = np.zeros((len(slen_list), len(alpha_list)))
        for i, slen in enumerate(slen_list):
            short_contexts = [x[-slen:] for x in full_contexts]
            short_logprobs = lm.loglikelihood(short_contexts, targets, sf_norm=False)

            for full, short, target in zip(full_logprobs, short_logprobs, targets):
                for j, alpha in enumerate(alpha_list):
                    ppl[i, j] += lm.contrast_continuation(full, short, target, alpha)
                    acc[i, j] += lm.greedy_matching(full, target, short, alpha)

        return (ppl, acc, len(batch))

    def aggregating(self, results, slen_list=[], alpha_list=[], **kwargs):
        # Aggregate results
        ppl = np.zeros((len(slen_list), len(alpha_list)))
        acc = np.zeros((len(slen_list), len(alpha_list)))
        total = 0

        for res in results:
            ppl += res[0]
            acc += res[1]
            total += res[2]

        ppl /= total
        acc /= total
        ppl = np.exp(-ppl)

        return (ppl, acc)

    def return_results(self, results, set_params={}, printing=True, slen_list=[], alpha_list=[], **kwargs):
        ppl, acc = results[0], results[1]
        slen_dict = {i: v for i, v in enumerate(slen_list)}
        alpha_dict = {i: v for i, v in enumerate(alpha_list)}
        ppl_best = np.unravel_index(np.argmax(ppl, axis=None), ppl.shape)
        acc_best = np.unravel_index(np.argmax(acc, axis=None), acc.shape)

        if printing:
            print('Task: {}, Model: {}, Description: {}, Few-shot: {}'.format(
                kwargs['task'], kwargs['model_name'], kwargs['add_description'], kwargs['num_shots']
            ))
            print('Accuracy')
            csv_writer = TsvTableWriter()
            table_header = 'SLength/Alpha'
            csv_writer.headers = [table_header] + alpha_list
            values = []
            for i in range(len(slen_list)):
                values.append([slen_list[i]] + list(acc[i]))

            csv_writer.value_matrix = values
            print(csv_writer.dumps())

        if 'alpha' in set_params and 'short_len' in set_params:
            short_len, alpha = set_params['short_len'], set_params['alpha']
            sl_idx, alpha_idx = slen_list.index(short_len), alpha_list.index(alpha)
            print("Acc={} ppl={} at short_len={}, alpha={}".format(acc[sl_idx, alpha_idx], ppl[sl_idx, alpha_idx], short_len, alpha))
            return {'acc': acc[sl_idx, alpha_idx], 
                    'ppl': ppl[sl_idx, alpha_idx], 
                    'short_len': short_len, 
                    'alpha': alpha, 
                    'acc_mat': acc,
                    'ppl_mat': ppl}
        else:
            print("Best acc={}, ppl={} at short_len={}, alpha={}".format(acc[acc_best], ppl[acc_best], slen_dict[acc_best[0]], alpha_dict[acc_best[1]]))
            return {'acc': acc[acc_best], 
                    'ppl': ppl[acc_best], 
                    'short_len': slen_dict[acc_best[0]], 
                    'alpha': alpha_dict[acc_best[1]],
                    'acc_mat': acc,
                    'ppl_mat': ppl}
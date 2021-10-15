import os
import sys
import time
import numpy as np

import torch
import torch.nn.functional as F
torch.set_grad_enabled(False)

import openai
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import *

def get_model(model_name):
    if 'gpt2' in model_name:
        return GPT2
    elif 'gpt3' in model_name:
        return GPT3
    else:
        raise Exception('Wrong model name.')


class GPT2():
    def __init__(self, model_name, device='cuda', **kwargs):
        super().__init__()
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        try:
            model_name = gpt2_map[model_name]
        except:
            raise Exception('Wrong model name pf gpt2!')

        self.model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id).to(self.device)
        self.model.eval()

        self.max_length = self.model.config.n_ctx

    def _model_call(self, inputs):
        return self.model(inputs)[0][:, :, :50257]

    def loglikelihood(self, contexts, targets, sf_norm=True):
        """Baisc unit to calculate logprobs for continuations (targets)

        Args:
            contexts ([type]): [description]
            targets ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Tokenization (when necessary)
        full_ids, tgt_ids, inp_lens = [], [], []
        
        for i in range(len(contexts)):
            cur_ctx, cur_tgt = contexts[i], targets[i]

            if isinstance(cur_ctx, str):
                cur_ctx = self.tokenizer(cur_ctx, verbose=False)['input_ids']
                cur_tgt = self.tokenizer(cur_tgt, verbose=False)['input_ids']
                
            inp = torch.tensor((cur_ctx + cur_tgt)[-(self.max_length+1):][:-1], dtype=torch.long)

            full_ids.append(inp.unsqueeze(0))
            inp_lens.append(inp.shape[0])
            tgt_ids.append(cur_tgt)

        # Padding
        max_len = max([x.shape[1] for x in full_ids])
        batch_fulls = []
        for s in full_ids:
            ss = torch.cat([s, torch.zeros(1, max_len - s.shape[1])], dim=1)
            batch_fulls.append(ss)
        batch_fulls = torch.cat(batch_fulls, dim=0).to(self.device, dtype=torch.long)  # [B, L]

        # LM Forward
        if sf_norm:
            full_logits = F.log_softmax(self._model_call(batch_fulls), dim=-1).cpu()  # [B, L, V]
        else:
            full_logits = self._model_call(batch_fulls).cpu()  # [B, L, V]

        # Gathering
        res = []
        for llogits, tgt_id, inp_len in zip(full_logits, tgt_ids, inp_lens):
            tgt_len = len(tgt_id)  # slen
            tgt_llogits = llogits[inp_len-tgt_len:inp_len].unsqueeze(0) # [1, L', V]
            res.append(tgt_llogits)
            
        return res

    def continuation_ll(self, logits_1, target, logits_2=None, lennorm=False):
        if isinstance(target, str):
            target = self.tokenizer.encode(target)

        tgt_id = torch.tensor(target, dtype=torch.long).unsqueeze(0)
        logits_1 = torch.gather(logits_1, 2, tgt_id.unsqueeze(-1)).squeeze(-1) # [1, L]
        l1 = float(logits_1.sum()) / logits_1.shape[1] if lennorm else float(logits_1.sum())

        if logits_2 is not None:
            logits_2 = torch.gather(logits_2, 2, tgt_id.unsqueeze(-1)).squeeze(-1) # [1, L]
            l2 = float(logits_2.sum()) / logits_2.shape[1] if lennorm else float(logits_2.sum())

            return l1, l2
        else:
            return l1

    def contrast_continuation(self, logits_1, logits_2, target, alpha=None, lennorm=False):
        if isinstance(target, str):
            target = self.tokenizer.encode(target)

        tgt_id = torch.tensor(target, dtype=torch.long).unsqueeze(0)

        if logits_2 is not None:
            logprobs = logprob_contrast(logits_1, logits_2, alpha)
            logprobs = F.log_softmax(logprobs, dim=-1)
        else:
            logprobs = F.log_softmax(logits_1, dim=-1)

        target_lp = torch.gather(logprobs, 2, tgt_id.unsqueeze(-1)).squeeze(-1) # [1, L]

        ll = float(target_lp.sum()) / target_lp.shape[1] if lennorm else float(target_lp.sum())

        return ll

    def greedy_matching(self, logits_1, target, logits_2=None, alpha=0):
        if isinstance(target, str):
            target = self.tokenizer.encode(target)

        target = torch.tensor(target, dtype=torch.long).unsqueeze(0)

        if logits_2 is not None:
            logits = logprob_contrast(logits_1, logits_2, alpha)
        else:
            logits = logits_1

        greedy_tokens = logits.argmax(dim=-1)
        max_equal = (greedy_tokens == target).all()
        return bool(max_equal)



class GPT3():
    def __init__(self, model_name, gpt3_apikey='', **kwargs):
        super().__init__()
        self.api_key = gpt3_apikey
        # self.api_key = open('api_key.txt').readlines()[0].strip()

        try:
            engine = gpt3_map[model_name]
        except:
            raise Exception('Wrong model name pf gpt3!')

        self.engine = engine
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.max_length = 2048
        openai.api_key = self.api_key

    def oa_completion(self, **kwargs):
        backoff_time = 3
        while True:
            try:
                return openai.Completion.create(**kwargs)
            except openai.error.OpenAIError:
                time.sleep(backoff_time)
                backoff_time *= 1.5

    def loglikelihood(self, contexts, targets, sf_norm=True):

        full_ids, ctx_lens = [], []
        for i in range(len(contexts)):
            cur_ctx, cur_tgt = contexts[i], targets[i]

            if isinstance(cur_ctx, str):
                cur_ctx, cur_tgt = self.tokenizer.encode(cur_ctx), self.tokenizer.encode(cur_tgt)

            inp = (cur_ctx + cur_tgt)[-self.max_length:]
            ctx_len = len(cur_ctx) - max(0, len(cur_ctx) + len(cur_tgt) - self.max_length)

            full_ids.append(inp)
            ctx_lens.append(ctx_len)

        # LM Forward
        responses = self.oa_completion(
                engine=self.engine,
                prompt=full_ids,
                echo=True,
                max_tokens=0, 
                temperature=0.,
                logprobs=100,
            )

        # Gathering
        res = []
        for response, ctx_len in zip(responses['choices'], ctx_lens):

            logprobs = response["logprobs"]
            res.append(logprobs)

        return res

    def continuation_ll(self, logits_1, target, logits_2=None, lennorm=False):
        if isinstance(target, str):
            target = self.tokenizer.encode(target)

        lp_1 = logits_1["token_logprobs"]
        cont_lp_1 = sum(lp_1[-len(target):]) / len(target) if lennorm else sum(lp_1[-len(target):])

        if logits_2 is not None:
            lp_2 = logits_2["token_logprobs"]
            cont_lp_2 = sum(lp_2[-len(target):]) / len(target) if lennorm else sum(lp_2[-len(target):])

            return cont_lp_1, cont_lp_2
        else:
            return cont_lp_1

    def contrast_continuation(self, logits_1, logits_2, target, alpha=None, lennorm=False):
        if isinstance(target, str):
            target = self.tokenizer.encode(target)

        lp_1 = logits_1["token_logprobs"]
        cont_lp_1 = sum(lp_1[-len(target):]) / len(target) if lennorm else sum(lp_1[-len(target):])

        if logits_2 is not None:
            lp_2 = logits_2["token_logprobs"]
            cont_lp_2 = sum(lp_2[-len(target):]) / len(target) if lennorm else sum(lp_2[-len(target):])

            return logprob_contrast(cont_lp_1, cont_lp_2, alpha)
        else:
            return cont_lp_1

    def greedy_matching(self, logits_1, target, logits_2=None, alpha=0):
        if isinstance(target, str):
            target = self.tokenizer.encode(target)

        if logits_2 is not None:
            is_greedy = True
            for i in range(-len(target), 0):
                token = logits_1["tokens"][i]

                logits_1 = logits_1['top_logprobs'][i]  # 100 * [word: logprob]
                logits_2 = logits_2['top_logprobs'][i]
                def_logit2 = np.log((1 - np.sum([np.exp(x) for x in logits_2.values()])) / (50257 - 100))
                
                diff_logits = {}
                for k in logits_1.keys():
                    if k in logits_2:
                        diff_logits[k] = logits_1[k] - alpha * logits_2[k]
                    else:
                        diff_logits[k] = logits_1[k] - alpha * def_logit2

                top_token = max(diff_logits.keys(), key=lambda x: diff_logits[x])

                if top_token != token:
                    is_greedy = False
                    break
                
            return is_greedy
        else:
            is_greedy = True
            for i in range(-len(target), 0):
                token = logits_1["tokens"][i]

                logits_1 = logits_1['top_logprobs'][i]  # 100 * [word: logprob]
                top_token = max(logits_1.keys(), key=lambda x: logits_1[x])

                if top_token != token:
                    is_greedy = False
                    break
                
            return is_greedy

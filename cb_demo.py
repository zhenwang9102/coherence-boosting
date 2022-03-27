import os
import sys
import argparse
import numpy as np
from tabulate import tabulate


import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
   

def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]


def arg_parser():
    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--context', type=str, default='gpt2-small')
    parser.add_argument('--short_length', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    return vars(args)


def print_ranking(logits, target, vocab_dict, top_k=5):
    sorted_ids = np.argsort(logits)[::-1]
    topk_ids = sorted_ids[:top_k]
    
    topk_tokens = [vocab_dict[id] + '**' if id == target else vocab_dict[id] for id in topk_ids]
    topk_logprobs = [logits[i] for i in topk_ids]
    topk_pct = [np.exp(logits[i]) * 100 for i in topk_ids]
    topk_rank = list(range(1, top_k+1))
    
    print_list = []
    for i in range(top_k):
        print_list.append(
            [
            "{}".format(topk_rank[i]),                   
            topk_tokens[i],
            "{:.3f}".format(topk_logprobs[i]),
            "{:.2f}%".format(topk_pct[i]),
            ]
        )

    if target not in topk_ids:
        target_idx = np.where(sorted_ids==target)[0][0]
        
        if target_idx != top_k:
            print_list += [['...'] * 4]

        print_list += [
            [
                "{}".format(target_idx + 1),
                vocab_dict[target] + '**',
                "{:.3f}".format(logits[target]),
                "{:.6f}%".format(np.exp(logits[target]) * 100),
            ]
        ]

    headers = ['Rank', 'Tokens', 'Logprobs', 'Probs']
    print(tabulate(print_list, headers=headers, floatfmt=".3f", tablefmt="simple", numalign="left"))
    print('** Target Token')
    print('\n')


def contrasting(model_name='gpt2', context='', partial_length=5, alpha=0.5, **kwargs):
    try:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    except RuntimeError:
        device = 'cpu'
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_dict = {id: token for token, id in tokenizer.get_vocab().items()}
    # print('Load the model {} to {}!'.format(model_name, device))
    
    input_ids = tokenizer.encode(context, return_tensors='pt').to(device)
    context = input_ids[:, :-1]
    target = input_ids[0, -1].item()
    
    long_logits = F.log_softmax(model(context)[0][:, -1, :].detach().cpu(), dim=1).numpy().reshape(-1)
    short_logits = F.log_softmax(model(context[:, -partial_length:])[0][:, -1, :].detach().cpu(), dim=1).numpy().reshape(-1)
    cb_logits = (1 + alpha) * long_logits - alpha * short_logits

    # print('Long context:\t{}\nShort context:\t{}\nTarget token:\t{}\n'.format(
    #     tokenizer.decode(context[0]),
    #     tokenizer.decode(context[0, -partial_length:]),
    #     tokenizer.decode(target)
    # ))
    
    print('Top tokens based on full context:\n{}\n'.format(color.BOLD + tokenizer.decode(context[0]) + color.END))
    print_ranking(long_logits, target, vocab_dict)

    print('Top tokens based on partial context:\n{}\n'.format(color.BOLD + tokenizer.decode(context[0, -partial_length:]) + color.END))
    print_ranking(short_logits, target, vocab_dict)

    print('Contrastive next token prediction:\n')
    print_ranking(cb_logits, target, vocab_dict)
    

if __name__ == '__main__':
    contrasting(**arg_parser())
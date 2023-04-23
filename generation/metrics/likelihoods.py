# Compute the long context sensitivity metrics (delta and LTF)
# Runs in ~50min on RTX8000 GPU
# Note that the results may vary slightly due to precision issues
#
# Example usage:
# python likelihoods.py ../outputs/conditional_topp_cb64.1.jsonl
# delta: 6.40
# LTF: 3.99

import json
import sys
import torch
import transformers
import tqdm
import numpy as np
import random

fn = sys.argv[1]

data = [ json.loads(l[:-1]) for l in open(fn, 'r') ]

torch.set_grad_enabled(False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
model = transformers.AutoModelForCausalLM.from_pretrained('gpt2-large').to(device)
model.eval()

long_batch_size = 64 # works with plenty of memory to spare (max ~22GB)

short_context_length = 20
ltf_thresholds = 0.05, 0.2

all_logits_long = []
all_logits_short = []

 # for impatient users, shuffle to get less biased estimators on partial results (since data is ordered by context length)
random.shuffle(data)

# compute likelihoods of the generated tokens, conditioned on the full context
# long likelihoods are batched across the dataset
# short likelihoods are one batch per example (uses max ~8GB GPU memory)
num_batches = int(np.ceil(len(data) / long_batch_size))

bar = tqdm.trange(num_batches)
for i_batch in bar:
    data_batch = data[i_batch * long_batch_size:(i_batch + 1) * long_batch_size]

    # long
    sequences, context_lengths, sequence_lengths = [], [], []
    for d in data_batch:
        context = torch.LongTensor(d['context']).to(device)
        generated = torch.LongTensor(d['tokens']).to(device)
        all_tokens = torch.cat([context, generated], 0)

        sequences.append(all_tokens)
        context_lengths.append(len(context))
        sequence_lengths.append(len(all_tokens))

    batch = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True).to(device)
    logits = model(batch)[0].log_softmax(-1)[:,:-1].gather(-1, batch[:,1:].unsqueeze(-1)).squeeze(-1)

    for lp, cl, sl in zip(logits, context_lengths, sequence_lengths):
        all_logits_long += list(lp[cl - 1:sl - 1].cpu().numpy())

    # short
    for d in data_batch:
        context = torch.LongTensor(d['context']).to(device)
        generated = torch.LongTensor(d['tokens']).to(device)
        all_tokens = torch.cat([context, generated], 0)

        windows, window_lengths, target_words = [], [], []
        for i in range(len(generated)):
            windows.append(all_tokens[max(0, i + len(context) - short_context_length):i + len(context) ])
            window_lengths.append(len(windows[-1]))
            target_words.append(all_tokens[i + len(context)])

        batch = torch.nn.utils.rnn.pad_sequence(windows, batch_first=True).to(device)
        logits = model(batch)[0].log_softmax(-1)[ torch.arange(len(windows)).to(device), 
                                            torch.LongTensor(window_lengths).to(device) - 1, 
                                            torch.LongTensor(target_words).to(device) ]
        all_logits_short += list(logits.cpu().numpy())

    all_probs_long = np.exp(np.array(all_logits_long))
    all_probs_short = np.exp(np.array(all_logits_short))

    delta = (all_probs_long - all_probs_short).mean()
    ltf = (( all_probs_short < ltf_thresholds[0] ) * ( all_probs_long > ltf_thresholds[1] )).mean()

    bar.set_postfix({'delta': f'{100 * delta : .2f}', 'LTF': f'{100 * ltf : .2f}'})
    
print(f'delta: {100 * delta : .2f}')
print(f'LTF: {100 * ltf : .2f}')
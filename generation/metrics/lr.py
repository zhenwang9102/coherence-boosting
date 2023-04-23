# Compute the long-range repetition metric (LR_50 and LR_100)
# Example usage:
# python lr.py ../outputs/conditional_topp_cb64.1.jsonl
# LR_50:  15.75
# LR_100:  9.67

import json
import sys

fn = sys.argv[1]

data = [ json.loads(l[:-1]) for l in open(fn, 'r') ]

lengths = [ 50, 100 ]
rn = { l: 0 for l in lengths }
s = 0

for d in data:
    tokens = d['tokens']
    distinct = set(tokens)
    
    s += len(distinct)

    for t in distinct:
        # find first and last occurrence of t
        first = tokens.index(t)
        last = len(tokens) - tokens[::-1].index(t) - 1
        for l in lengths:
            if last - first >= l:
                rn[l] += 1

for l in lengths:
    print(f'LR_{l}: {100 * (rn[l] / s) : .2f}')
    
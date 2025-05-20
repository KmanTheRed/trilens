#!/usr/bin/env python3
import json
import random
import argparse

def sample_balanced(input_path, output_path, n_per_class, seed=42):
    random.seed(seed)
    human_res = []    # reservoir for label=0
    ai_res    = []    # reservoir for label=1
    human_cnt = 0
    ai_cnt    = 0

    with open(input_path, encoding='utf8') as f:
        for line in f:
            obj = json.loads(line)
            lbl = obj.get('label')
            if lbl == 0:
                human_cnt += 1
                if len(human_res) < n_per_class:
                    human_res.append(line)
                else:
                    j = random.randrange(human_cnt)
                    if j < n_per_class:
                        human_res[j] = line
            elif lbl == 1:
                ai_cnt += 1
                if len(ai_res) < n_per_class:
                    ai_res.append(line)
                else:
                    j = random.randrange(ai_cnt)
                    if j < n_per_class:
                        ai_res[j] = line

    if human_cnt < n_per_class or ai_cnt < n_per_class:
        raise ValueError(f"Not enough examples: {human_cnt} human, {ai_cnt} AI")

    with open(output_path, 'w', encoding='utf8') as out:
        for l in human_res + ai_res:
            out.write(l)

    print(f"Done: sampled {len(human_res)} human and {len(ai_res)} AI lines to {output_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Reservoir‐sample a balanced JSONL")
    p.add_argument('--input',       required=True, help='path to data/raw/train.jsonl')
    p.add_argument('--output',      required=True, help='where to write the balanced JSONL')
    p.add_argument('--n_per_class', type=int, default=12500, help='number of examples per class')
    p.add_argument('--seed',        type=int, default=42)
    args = p.parse_args()
    sample_balanced(args.input, args.output, args.n_per_class, args.seed)

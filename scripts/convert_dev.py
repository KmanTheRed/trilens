#!/usr/bin/env python3
import json
import argparse

def convert(input_path, output_path):
    """
    Reads a JSONL where each obj has:
      { 'human_text': str, 'machine_text': str, ... }
    and writes out one line per text with { 'text':…, 'label':… }.
    """
    cnt_h, cnt_a = 0, 0
    with open(input_path, encoding='utf8') as fin, \
         open(output_path, 'w', encoding='utf8') as fout:

        for line in fin:
            obj = json.loads(line)
            # human example
            h = obj.get('human_text')
            if isinstance(h, str):
                fout.write(json.dumps({'text': h, 'label': 0}) + '\n')
                cnt_h += 1
            # AI example
            a = obj.get('machine_text')
            if isinstance(a, str):
                fout.write(json.dumps({'text': a, 'label': 1}) + '\n')
                cnt_a += 1

    print(f"Converted dev: {cnt_h} human, {cnt_a} AI → {cnt_h+cnt_a} total lines")

if __name__ == '__main__':
    p = argparse.ArgumentParser(prog="convert_dev.py")
    p.add_argument('--input',  required=True, help='raw dev JSONL')
    p.add_argument('--output', required=True, help='standardized JSONL')
    args = p.parse_args()
    convert(args.input, args.output)

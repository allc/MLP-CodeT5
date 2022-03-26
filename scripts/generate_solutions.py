import sys
import os
import argparse
import json
import random
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
sys.path.append(os.path.abspath('../'))
from _utils import build_codecontest_input


LANGUAGES = [
    'python3',
]
MAX_LENGTH = 2048


'''
Output JSONL:
{
    'problem_name': '',
    'generated_solutions': [
        'pseudo_rating': 800,
        'pseudo_tags': [],
        'language': 'python',
        'code': ''
    ]
}
'''


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('in_file')
    argparser.add_argument('out_dir')
    argparser.add_argument('--model', choices=['base', 'small'], default='base')
    argparser.add_argument('--model_dir')
    argparser.add_argument('--metadata_sample', type=int, default=50)
    argparser.add_argument('--sample_num', type=int, default=20)
    argparser.add_argument('--continue', action='store_true', dest='is_continued')
    argparser.add_argument('--no_cuda', action='store_true')
    args = argparser.parse_args()

    if args.model == 'base':
        tokenizer_path = 'Salesforce/codet5-base'
        model_path = 'Salesforce/codet5-base'
    elif args.model == 'small':
        tokenizer_path = 'Salesforce/codet5-small'
        model_path = 'Salesforce/codet5-small'
    if 'model_dir' in args:
        model_path = args.model_dir
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    if args.no_cuda or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'
    model.to(device)
    print('Model loaded')

    most_popular_tags = get_most_popular_tags(os.path.join(os.path.dirname(args.in_file), 'code_contests_train_most_popular_tags.txt'))

    out_file = os.path.join(args.out_dir, 'generated_solutions.jsonl')
    skip_lines = 0
    if args.is_continued:
        with open(out_file) as f:
            skip_lines = sum(1 for _ in f)
            print(f'{skip_lines} problems will be skipped')
    
    with open(args.in_file) as f:
        file_mode = 'a' if args.is_continued else 'w'
        out_f = open(out_file, file_mode)
        ln = 0
        for line in f:
            ln += 1
            if ln <= skip_lines:
                continue
            x = json.loads(line)
            out_data = {'problem_name': x['problem_name'], 'generated_solutions': []}
            for i in range(args.metadata_sample):
                print(f'\rSolving problem {ln} with metadata sample {i}', end='')
                rating = random.randint(8, 35) * 100
                tags = random.choice(most_popular_tags)
                language = random.choice(LANGUAGES)
                nl = build_codecontest_input(rating, tags, language, True, x['problem_description'])
                input_ids = tokenizer(nl, return_tensors="pt").input_ids
                input_ids = input_ids.to(device)
                generated_ids = model.generate(
                    input_ids,
                    do_sample=True,
                    max_length=MAX_LENGTH,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=args.sample_num,
                )
                generated_solutions = map(lambda x: tokenizer.decode(x, skip_special_tokens=True), generated_ids)
                for s in generated_solutions:
                    out_data['generated_solutions'].append({'pseudo_rating': rating, 'pseudo_tags': tags, 'language': language, 'code': s})
                if device == 'cuda':
                    torch.cuda.empty_cache()
            out_f.write(json.dumps(out_data))
            out_f.write('\n')
        print('\n')
        out_f.flush()
        out_f.close()



def get_most_popular_tags(file, n=50):
    most_popular_tags = []
    with open(file) as f:
        for _ in range(n):
            line = f.readline()
            if line == '':
                break
            line = line[:-1]
            most_popular_tags.append(line.split(', '))
    return most_popular_tags


if __name__ == '__main__':
    main()

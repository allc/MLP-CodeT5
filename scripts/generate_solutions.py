import sys
import os
import json
import random
from transformers import RobertaTokenizer, T5ForConditionalGeneration
sys.path.append(os.path.abspath('../'))
from _utils import build_codecontest_input


LANGUAGES = [
    'python',
]
METADATA_SAMPLE_TIMES = 10
NUMBER_OF_GENERATE_SAMPLE = 10
MAX_LENGTH = 2048
DEVICE='cuda'


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
    in_file = sys.argv[1]
    out_dir = sys.argv[2]

    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
    model.to(DEVICE)
    print('Model loaded')

    most_popular_tags = get_most_popular_tags(os.path.join(os.path.dirname(in_file), 'code_contests_train_most_popular_tags.txt'))
    
    with open(in_file) as f:
        out_f = open(os.path.join(out_dir, 'generated_solutions.jsonl'), 'w')
        ln = 0
        for line in f:
            ln += 1
            x = json.loads(line)
            out_data = {'problem_name': x['problem_name'], 'generated_solutions': []}
            for i in range(METADATA_SAMPLE_TIMES):
                print(f'\rSolving problem {ln} with metadata sample {i}', end='')
                rating = random.randint(8, 35) * 100
                tags = random.choice(most_popular_tags)
                language = random.choice(LANGUAGES)
                nl = build_codecontest_input(rating, tags, language, True, x['problem_description'])
                input_ids = tokenizer(nl, return_tensors="pt").input_ids
                input_ids = input_ids.to(DEVICE)
                generated_ids = model.generate(
                    input_ids,
                    do_sample=True,
                    max_length=MAX_LENGTH,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=NUMBER_OF_GENERATE_SAMPLE,
                )
                generated_solutions = map(lambda x: tokenizer.decode(x, skip_special_tokens=True), generated_ids)
                for s in generated_solutions:
                    out_data['generated_solutions'].append({'pseudo_rating': rating, 'pseudo_tags': tags, 'language': language, 'code': s})
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

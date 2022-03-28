import sys
import json


def main():
    in_file = sys.argv[1]
    with open(in_file) as f:
        out_data = {'problem_name': '', 'generated_solutions': []}
        for line in f:
            x = json.loads(line)
            if out_data['problem_name'] == '' or x['problem_name'] == out_data['problem_name']:
                out_data['generated_solutions'] += x['generated_solutions']
            else:
                print(json.dumps(out_data))
                out_data = x
        print(json.dumps(out_data))


if __name__ == '__main__':
    main()
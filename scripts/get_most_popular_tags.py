import sys
import os
import json
from collections import defaultdict

def main():
    in_file = sys.argv[1]
    tags_count = defaultdict(lambda: 0)
    with open(in_file) as f:
        for line in f:
            x = json.loads(line)
            tags = sorted(x['tags'])
            tags = ', '.join(tags)
            tags_count[tags] += 1
    out_file = os.path.splitext(in_file)[0] + '_most_popular_tags.txt'
    with open(out_file, 'w') as f:
        tags = sorted(tags_count.keys(), key=lambda tgs: tags_count[tgs], reverse=True)
        f.write('\n'.join(tags))


if __name__ == '__main__':
    main()

import sys
from CWSTokenizer import CWSTokenizer

input_file = sys.argv[1]
output_file = sys.argv[2]

t = CWSTokenizer()
lines = [line.strip('\n') for line in open(input_file, 'r')]
tokenized_lines = [t.sent_tokenize(line)[1] for line in lines]
tokenized_lines = [
    ' '.join([''.join(segment) for segment in line]) for line in tokenized_lines
]

with open(output_file, 'w+') as f:
    for line in tokenized_lines:
        print(line, file=f)

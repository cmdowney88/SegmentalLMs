import sys
import random
import math

name = sys.argv[1]
train = float(sys.argv[2])
dev = float(sys.argv[3])

assert (0 < train < 1)
assert (0 < dev < 1)
assert (0 < train + dev < 1)

random.seed(95)

lines = [line.rstrip('\n') for line in sys.stdin]
random.shuffle(lines)

train_index = math.ceil(len(lines) * train)
dev_index = math.ceil(len(lines) * (train + dev))

train_lines = lines[:train_index]
dev_lines = lines[train_index:dev_index]
test_lines = lines[dev_index:]

assert (len(train_lines) + len(dev_lines) + len(test_lines) == len(lines))

with open(name + '_train.txt', 'w') as f:
    for line in train_lines:
        print(line, file=f)

with open(name + '_dev.txt', 'w') as f:
    for line in dev_lines:
        print(line, file=f)

with open(name + '_test.txt', 'w') as f:
    for line in test_lines:
        print(line, file=f)

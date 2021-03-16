import sys
import json
import chonker.wrangle as wr
import numpy as np
from gensim.models import Word2Vec

input_file = sys.argv[1]
window_size = int(sys.argv[2])
embedding_size = int(sys.argv[3])
num_epochs = int(sys.argv[4])
output_file = sys.argv[5]

lines = wr.character_tokenize(input_file)

model = Word2Vec(min_count=1, window=window_size, size=embedding_size, workers=4)
model.build_vocab(lines)
model.train(lines, total_examples=model.corpus_count, epochs=num_epochs)

vocab = {}
vectors = []
for i, item in enumerate(model.wv.index2word):
    vocab[item] = i
    vectors.append(model.wv[item])
vectors = np.array(vectors)

np.save(f"{output_file}.npy", vectors)
with open(f"{output_file}_indices.json", "w+") as f:
    json.dump(vocab, f)

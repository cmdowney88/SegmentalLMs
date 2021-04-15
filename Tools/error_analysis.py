import argparse
import chonker.wrangle as wr
import numpy as np
from typing import List
from sklearn.metrics import precision_recall_fscore_support as fscore


def get_boundary_vector(sequence: List[List[str]]) -> List[int]:
    lengths = [len(segment) for segment in sequence]
    num_symbols = sum(lengths)
    vector = [0 for x in range(num_symbols)]
    current_index = 0
    for length in lengths[:-1]:
        current_index += length
        vector[current_index - 1] = 1
    return vector[:-1]


def main():
    parser = argparse.ArgumentParser(
        description='Script to get basic segmentation comparison to gold file'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        required=True,
        help='File containing the test segmentation'
    )
    parser.add_argument(
        '--gold_file',
        type=str,
        required=True,
        help='File containing the gold segmentation'
    )
    args = parser.parse_args()

    test_words = wr.basic_tokenize(
        args.test_file, preserve_case=False, split_tags=True
    )
    num_test_words = len([word for word in wr.flatten(test_words) if word != ''])
    test_text = [wr.chars_from_words(sent) for sent in test_words]
    test_boundaries = [get_boundary_vector(ex) for ex in test_text]
    all_test_boundaries = np.array(wr.flatten(test_boundaries))
    num_test_chars = len(all_test_boundaries) + len(test_boundaries)

    gold_words = wr.basic_tokenize(
        args.gold_file, preserve_case=False, split_tags=True
    )
    num_gold_words = len([word for word in wr.flatten(gold_words) if word != ''])
    gold_text = [wr.chars_from_words(sent) for sent in gold_words]
    gold_boundaries = [get_boundary_vector(ex) for ex in gold_text]
    all_gold_boundaries = np.array(wr.flatten(gold_boundaries))
    num_gold_chars = len(all_gold_boundaries) + len(gold_boundaries)

    assert num_test_chars == num_gold_chars

    test_word_length = round(num_test_chars / num_test_words, 2)
    gold_word_length = round(num_gold_chars / num_gold_words, 2)

    precision, recall, f1, _ = fscore(
        all_gold_boundaries, all_test_boundaries, average='binary'
    )

    print(f"Characters: {num_gold_chars}")
    print(f"Gold words: {num_gold_words}")
    print(f"Test words: {num_test_words}")
    print(f"Average gold word length: {gold_word_length}")
    print(f"Average test word length: {test_word_length}")
    print(f"Test F1: {round(100*f1, 1)}")
    print(f"Precision: {round(100*precision, 1)}")
    print(f"Recall: {round(100*recall, 1)}")


if __name__ == "__main__":
    main()

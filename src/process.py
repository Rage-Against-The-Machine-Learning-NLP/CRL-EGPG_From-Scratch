import os
from typing import Dict, List
import pickle as pkl

import nltk

from utils import Specials


def create_vocab(
    store_validation: bool = True, min_freq: int = 2, data_path: str = "./data/para"
) -> None:
    word_to_count: Dict[str, int] = {}

    def process_file(file_path: str) -> None:
        with open(file_path, "r") as file:
            for line in file.readlines():
                words = [word.lower() for word in nltk.word_tokenize(line)]
                tags = list(zip(*nltk.pos_tag(words)))[1]

                for word in words + tags:
                    word_to_count[word] = word_to_count.get(word, 0) + 1

    files = ["train_src.txt", "train_trg.txt"]
    if store_validation:
        files += ["val_src.txt", "val_trg.txt"]

    for file in files:
        process_file(os.path.join(data_path, file))

    specials: List[str] = [Specials.PAD, Specials.UNK, Specials.SOS, Specials.EOS]

    word_to_index: Dict[str, int] = {word: index for index, word in enumerate(specials)}
    for word, count in word_to_count.items():
        if count < min_freq:
            continue
        word_to_index[word] = len(word_to_index)

    index_to_word = {index: word for word, index in word_to_index.items()}

    data_out_path = os.path.join(data_path, "processed")
    os.makedirs(data_out_path, exist_ok=True)

    def dump_data(data, file_path: str) -> None:
        with open(file_path, "wb") as file:
            pkl.dump(data, file)

    dump_data(word_to_index, os.path.join(data_out_path, "word_to_index.pkl"))
    dump_data(index_to_word, os.path.join(data_out_path, "index_to_word.pkl"))

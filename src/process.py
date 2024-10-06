import os
from typing import Dict, List, cast, Tuple

import nltk

from utils import Specials, pkl_dump, pkl_load


def get_tokens(line: str) -> List[str]:
    return [word.lower() for word in nltk.word_tokenize(line)]


def create_vocab(
    store_validation: bool = True,
    min_freq: int = 2,
    data_path: str = "./data/para",
    cleaned_dir_name: str = "processed",
) -> None:
    word_to_count: Dict[str, int] = {}

    def process_file(file_path: str) -> None:
        with open(file_path, "r") as file:
            for line in file.readlines():
                words = get_tokens(line)
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
        if count < min_freq or word in word_to_index:
            continue
        word_to_index[word] = len(word_to_index)

    index_to_word = {index: word for word, index in word_to_index.items()}

    data_out_path = os.path.join(data_path, cleaned_dir_name)
    os.makedirs(data_out_path, exist_ok=True)

    pkl_dump(word_to_index, os.path.join(data_out_path, "word_to_index.pkl"))
    pkl_dump(index_to_word, os.path.join(data_out_path, "index_to_word.pkl"))


def token_to_index(
    data_path: str = "./data/para",
    sub_file: str = "train",
    cleaned_dir_name: str = "processed",
) -> None:
    data_out_path = os.path.join(data_path, cleaned_dir_name)

    w2i_data = pkl_load(os.path.join(data_out_path, "word_to_index.pkl"))
    word_to_index = cast(Dict[str, int], w2i_data)

    def process_file(file_path: str) -> Tuple[List[List[int]], List[List[int]]]:
        token_indices: List[List[int]] = []
        tag_indices: List[List[int]] = []

        with open(file_path, "r") as file:
            for line in file.readlines():
                tokens = get_tokens(line)
                line_token_idx = [
                    word_to_index.get(word, word_to_index[Specials.UNK])
                    for word in tokens
                ]

                tags: List[str] = list(zip(*nltk.pos_tag(tokens)))[1]
                line_tag_idx = [
                    word_to_index.get(tag, word_to_index[Specials.UNK]) for tag in tags
                ]

                token_indices.append(line_token_idx)
                tag_indices.append(line_tag_idx)

        return token_indices, tag_indices

    src_tokens, src_tags = process_file(os.path.join(data_path, f"{sub_file}_src.txt"))
    trg_tokens, trg_tags = process_file(os.path.join(data_path, f"{sub_file}_trg.txt"))

    pkl_dump(src_tokens, os.path.join(data_out_path, f"{sub_file}_src.pkl"))
    pkl_dump(src_tags, os.path.join(data_out_path, f"{sub_file}_src_tags.pkl"))
    pkl_dump(trg_tokens, os.path.join(data_out_path, f"{sub_file}_trg.pkl"))
    pkl_dump(trg_tags, os.path.join(data_out_path, f"{sub_file}_trg_tags.pkl"))

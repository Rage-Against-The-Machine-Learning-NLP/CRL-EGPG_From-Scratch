import os
import time
from typing import Dict, List, cast, Tuple
import random
import multiprocessing

import numpy as np
import nltk
import editdistance
import argparse

from .utils import Specials, get_bert_tokenizer, pkl_dump, pkl_load

nltk.download("averaged_perceptron_tagger_eng")


def get_tokens(line: str) -> List[str]:
    """
    tokenize a line of text
    """
    return [word.lower() for word in nltk.word_tokenize(line)]


def store_vocab(
    store_validation: bool = True,
    min_freq: int = 2,
    data_path: str = "./data/para",
    cleaned_dir_name: str = "processed",
) -> None:
    """
    store the vocabulary for the given dataset, in the form of
    word_to_index and index_to_word mappings
    """
    word_to_count: Dict[str, int] = {}

    def process_file(file_path: str) -> None:
        with open(file_path, "r") as file:
            for line in file.readlines():
                words = get_tokens(line)
                tags = list(zip(*nltk.pos_tag(words)))[1]

                for word in words + list(tags):
                    word_to_count[word] = word_to_count.get(word, 0) + 1

    files = ["train_src.txt", "train_trg.txt"]
    if store_validation:
        files += ["valid_src.txt", "valid_trg.txt"]

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


def store_indices(
    data_path: str = "./data/para",
    sub_file: str = "train",
    cleaned_dir_name: str = "processed",
) -> None:
    """
    store the indices for each line of the sub_file in the data_path
    corresponding to the vocabulary
    """
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


def store_bert_ids(
    data_path: str = "./data/para",
    cleaned_dir_name: str = "processed",
    sub_file: str = "train",
) -> None:
    """
    store the bert ids for each line of the sub_file in the data_path
    corresponding to the vocabulary
    """
    data_out_path = os.path.join(data_path, cleaned_dir_name)
    i2w_data = pkl_load(os.path.join(data_out_path, "index_to_word.pkl"))
    src_token_data = pkl_load(os.path.join(data_out_path, f"{sub_file}_src.pkl"))
    trg_token_data = pkl_load(os.path.join(data_out_path, f"{sub_file}_trg.pkl"))

    index_to_word = cast(Dict[int, str], i2w_data)
    src_tokens = cast(List[List[int]], src_token_data)
    trg_tokens = cast(List[List[int]], trg_token_data)

    tokenizer = get_bert_tokenizer()

    def get_bert_ids(tokens: List[List[int]]) -> List[List[int]]:
        text = [" ".join([index_to_word[idx] for idx in line]) for line in tokens]
        return [tokenizer.encode(line, add_special_tokens=True) for line in text]

    src_bert_ids = get_bert_ids(src_tokens)
    trg_bert_ids = get_bert_ids(trg_tokens)

    pkl_dump(src_bert_ids, os.path.join(data_out_path, f"{sub_file}_src_bert_ids.pkl"))
    pkl_dump(trg_bert_ids, os.path.join(data_out_path, f"{sub_file}_trg_bert_ids.pkl"))


def _get_similar_sents(
    i: int, NUM_LINES: int, tokens: List[List[int]], tags: List[List[int]]
) -> Tuple[int, List[int]]:
    similarity = [np.inf for _ in range(NUM_LINES)]

    for j in range(NUM_LINES):
        if i == j:
            continue

        if abs(len(tokens[i]) - len(tokens[j])) > 2:
            continue  # if the sentence lengths are not close enough
        if len(list(set(tokens[i]))) - len(list(set(tokens[i]) & set(tokens[j]))) < 2:
            continue  # if the sentences don't have enough unique words in common

        similarity[j] = editdistance.eval(tags[i], tags[j])

    best_score = min(similarity)
    most_similar = [idx for idx, score in enumerate(similarity) if score == best_score]

    return i, random.sample(most_similar, min(5, len(most_similar)))


def store_similar_sents(
    data_path: str = "./data/para",
    cleaned_dir_name: str = "processed",
    sub_file: str = "test",
) -> None:
    """
    store the indices of similar sentences for each line of the sub_file in the data_path
    We only do this for the target sentences, as we have to guide their style
    similarity is measured by the edit distance of the PoS tags
    """
    data_out_path = os.path.join(data_path, cleaned_dir_name)
    tag_data = pkl_load(os.path.join(data_out_path, f"{sub_file}_trg_tags.pkl"))
    tok_data = pkl_load(os.path.join(data_out_path, f"{sub_file}_trg.pkl"))

    tags = cast(List[List[int]], tag_data)
    tokens = cast(List[List[int]], tok_data)

    NUM_LINES = len(tokens)

    with multiprocessing.Pool() as pool:
        from functools import partial

        partial_get_similar_sents = partial(
            _get_similar_sents, NUM_LINES=NUM_LINES, tokens=tokens, tags=tags
        )
        results = pool.map(partial_get_similar_sents, range(NUM_LINES))

    similar_list = [[] for _ in range(NUM_LINES)]
    for i, similar in results:
        similar_list[i] = similar

    pkl_dump(similar_list, os.path.join(data_out_path, f"{sub_file}_similarity.pkl"))


def _store_indices_wrapper(sub_file: str, data_path: str, event) -> None:
    """
    wrapper to store indices for a sub_file
    """
    store_indices(sub_file=sub_file, data_path=data_path)
    print(f"\t\t-> stored indices for {sub_file}")
    event.set()  # signal that indices are stored


def _store_bert_ids_wrapper(sub_file: str, data_path: str, event) -> None:
    """
    wrapper to store bert ids for a sub_file
    """
    event.wait()  # wait for indices to be stored
    store_bert_ids(sub_file=sub_file, data_path=data_path)
    print(f"\t\t-> stored bert ids for {sub_file}")


def _store_similar_sents_wrapper(sub_file: str, data_path: str, event) -> None:
    """
    wrapper to store similar sentences for a sub_file
    """
    event.wait()  # wait for indices to be stored
    store_similar_sents(sub_file=sub_file, data_path=data_path)
    print(f"\t\t-> stored similar sentences for {sub_file}")


def process_sub_file(sub_file: str, dataset_path: str) -> None:
    """
    store indices, bert ids and similar sentences for a sub_file
    """
    print(f"\tprocessing {sub_file}")

    event = multiprocessing.Event()

    indices_process = multiprocessing.Process(
        target=_store_indices_wrapper, args=(sub_file, dataset_path, event)
    )

    bert_process = multiprocessing.Process(
        target=_store_bert_ids_wrapper, args=(sub_file, dataset_path, event)
    )
    similar_process = multiprocessing.Process(
        target=_store_similar_sents_wrapper, args=(sub_file, dataset_path, event)
    )

    indices_process.start()
    bert_process.start()
    similar_process.start()

    indices_process.join()
    bert_process.join()
    similar_process.join()

    print(f"\tfinished processing {sub_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="process dataset info")
    parser.add_argument(
        "--dataset", type=str, required=True, help="dataset name (para / quora)"
    )

    args = parser.parse_args()
    dataset = args.dataset
    if dataset not in ["para", "quora"]:
        raise ValueError("invalid dataset name, must be 'para' or 'quora'")

    dataset_path = os.path.join("./data", dataset)
    print("dataset: ", dataset)

    store_vocab(data_path=dataset_path)
    print("\t-> stored vocab\n")

    sub_files = ["test", "valid", "train"]

    start_time = time.time()
    processes = []
    for sub_file in sub_files:
        p = multiprocessing.Process(
            target=process_sub_file, args=(sub_file, dataset_path)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    time_taken = time.time() - start_time
    print(f"processing complete in {time_taken}s")


# def show_sim() -> None:
#     """
#     show source, target-similar sentences and target sentence for a random line
#     """
#     data_path = "./data/quora"
#     cleaned_dir_name = "processed"
#     sub_file = "test"

#     data_out_path = os.path.join(data_path, cleaned_dir_name)

#     similar_data = pkl_load(os.path.join(data_out_path, f"{sub_file}_similarity.pkl"))
#     tok_data_trg = pkl_load(os.path.join(data_out_path, f"{sub_file}_trg.pkl"))
#     tok_data_src = pkl_load(os.path.join(data_out_path, f"{sub_file}_src.pkl"))
#     tag_data_trg = pkl_load(os.path.join(data_out_path, f"{sub_file}_trg_tags.pkl"))
#     index_to_word = pkl_load(os.path.join(data_out_path, "index_to_word.pkl"))

#     similar_list = cast(List[List[int]], similar_data)
#     tokens_trg = cast(List[List[int]], tok_data_trg)
#     tokens_src = cast(List[List[int]], tok_data_src)
#     tags_trg = cast(List[List[int]], tag_data_trg)
#     index_to_word = cast(Dict[int, str], index_to_word)

#     # show for a random line
#     idx = random.randint(0, len(tokens_trg))

#     print("SOURCE SENTENCE: ")
#     sentence = " ".join([index_to_word[idx] for idx in tokens_src[idx]])
#     print(f"\tSentence: {sentence}")
#     print("\n")

#     print("SIMILAR SENTENCES (TO TARGET STYLE BY PoS): ")
#     for sim_idx in similar_list[idx]:
#         sim_sentence = " ".join([index_to_word[idx] for idx in tokens_trg[sim_idx]])
#         sim_tag_list = [index_to_word[idx] for idx in tags_trg[sim_idx]]
#         print(f"\tSentence: {sim_sentence}")
#         print(f"\tTags: {sim_tag_list}")
#         print("\n")

#     sentence = " ".join([index_to_word[idx] for idx in tokens_trg[idx]])
#     tag_list = [index_to_word[idx] for idx in tags_trg[idx]]
#     print("TARGET SENTENCE: ")
#     print(f"\tSentence: {sentence}")
#     print(f"\tTags: {tag_list}")


if __name__ == "__main__":
    main()

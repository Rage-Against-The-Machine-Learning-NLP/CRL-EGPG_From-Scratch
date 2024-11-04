from dataclasses import dataclass
import pickle as pkl
import os, sys
import numpy as np

from transformers import BertTokenizer


def get_bert_tokenizer() -> BertTokenizer:
    return BertTokenizer.from_pretrained("bert-base-uncased")


@dataclass
class Specials:
    PAD: str = "<PAD>"
    UNK: str = "<UNK>"
    SOS: str = "<SOS>"
    EOS: str = "<EOS>"


def pkl_dump(data, file_path: str) -> None:
    with open(file_path, "wb") as file:
        pkl.dump(data, file)


def pkl_load(file_path: str):
    with open(file_path, "rb") as file:
        return pkl.load(file)


def resolve_relpath(relpath: str) -> str:
    dirname = os.path.dirname(__file__)
    abspath = os.path.join(dirname, relpath)
    sys.path.append(abspath)
    return abspath


def load_glove_model(filepath: str) -> dict[str, np.ndarray]:
    print("Loading Glove Model")

    with open(filepath, "r") as infile:
        gloveModel: dict[str, np.ndarray] = dict()
        for line in infile:
            splitLines = line.split()
            word: str = splitLines[0]
            wordEmbedding: np.ndarray = np.array(
                [float(value) for value in splitLines[1:]]
            )
            gloveModel[word] = wordEmbedding
        print(len(gloveModel), " words loaded!")

    return gloveModel


def initialise_word_embedding(glovefile: str, vocabfile: str) -> np.ndarray:
    with open(vocabfile, "rb") as f:
        vocab: dict[str, int] = pkl_load(f)

    glove_emb: dict[str, np.ndarray] = load_glove_model(glovefile)
    word_emb: np.ndarray = np.zeros((len(vocab), 300))
    missing_num = 0

    for word, idx in vocab.items():
        if word in glove_emb:
            word_emb[idx] = glove_emb[word]
            continue
        missing_num += 1

    print(str(missing_num) + " words are not in the glove embedding")
    return word_emb

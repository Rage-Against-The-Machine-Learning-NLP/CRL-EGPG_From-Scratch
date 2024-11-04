from dataclasses import dataclass
import pickle as pkl
import os, sys

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
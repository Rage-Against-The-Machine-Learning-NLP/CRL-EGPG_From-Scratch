from dataclasses import dataclass


@dataclass
class Specials:
    PAD: str = "<PAD>"
    UNK: str = "<UNK>"
    SOS: str = "<SOS>"
    EOS: str = "<EOS>"

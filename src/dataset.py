import os
import random
from typing import List
from utils import pkl_load

from torch.utils.data import Dataset
import torch


class OurDataset(Dataset):
    def __init__(self, dataroot: str, max_len: int, type: str):
        def resolve_paths() -> List[str]:
            """
            returns absolute paths of all files associated with the dataset (train/test/val)
            resolved against the root path provided
            """

            files = (
                "idx_to_word",
                "word_to_idx",
                "src",
                "src_bert_ids",
                "src_tags",
                "trg",
                "trg_bert_ids",
                "trg_tags",
                "similarity"
            )

            abspaths = []
            for file in files:
                is_vocab_file = file in ("idx_to_word", "word_to_idx")
                relpath = dataroot + "/" + (f"{self.type}_" if is_vocab_file else "") + file + ".pkl"
                abspaths.append(os.path.join(relpath))
            return abspaths
        

        super().__init__()

        assert type in ["train", "test", "val"]
        self.type = type
        self.max_len: int = max_len

        abspaths = resolve_paths()

        self.idx_to_word = pkl_load(abspaths[0])
        self.word_to_idx = pkl_load(abspaths[1])
        self.src = pkl_load(abspaths[2])
        self.src_bert_ids = pkl_load(abspaths[3])
        self.src_tags = pkl_load(abspaths[4])
        self.trg = pkl_load(abspaths[5])
        self.trg_bert_ids = pkl_load(abspaths[6])
        self.trg_tags = pkl_load(abspaths[7])
        self.similarity = pkl_load(abspaths[8])


    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int):
        """
        returns the source, target sentences, and a randomly chosen exemplar
        after trimming them according to the maximum allowed length
        """
        
        def trim_tensor(data, max_len: int, prepend=None, append=None) -> torch.Tensor:
            num_extra = int(prepend is not None) + int(append is not None)
            trunc = torch.zeros(max_len + num_extra, dtype=torch.long)
            length = min(len(data), max_len)
            
            start = int(prepend is not None)
            trunc[start: start + length] = torch.tensor(data[0: length])
            
            if prepend is not None:
                trunc[0] = prepend
            if append is not None:
                trunc[-1] = append
            
            return trunc

        # seq2seq data_quora format
        src_sent = self.src[idx]
        trg_sent = self.trg[idx]
        # src_len = len(src_sent)
        # trg_len = len(trg_sent)

        # src, src_len = trim_tensor(src_sent, self.max_len)
        # content_trg, content_len = trim_tensor(trg_sent, self.max_len)
        # trg, trg_len = trim_tensor(trg_sent, self.max_len, append=3) # EOS
        # trg_input, trg_len = trim_tensor(trg_sent, self.max_len, prepend=2)

        # # bert data_quora format
        # bert_in = self.src_bert_ids[idx]
        # bert_out = self.trg_bert_ids[idx]
        # exmp = self.trg_bert_ids[random.choice(self.similarity[idx])]
        # bert_src, _ = trim_tensor(bert_in, self.max_len + 2)
        # bert_trg, _ = trim_tensor(bert_out, self.max_len + 2)
        # bert_exmp, _ = trim_tensor(exmp, self.max_len + 2)

        src = trim_tensor(src_sent, self.max_len)
        content_trg = trim_tensor(trg_sent, self.max_len)
        trg = trim_tensor(trg_sent, self.max_len, append=3) # EOS
        trg_input = trim_tensor(trg_sent, self.max_len, prepend=2)

        # bert data_quora format
        bert_in = self.src_bert_ids[idx]
        bert_out = self.trg_bert_ids[idx]
        exmp = self.trg_bert_ids[random.choice(self.similarity[idx])]
        bert_src = trim_tensor(bert_in, self.max_len + 2)
        bert_trg = trim_tensor(bert_out, self.max_len + 2)
        bert_exmp = trim_tensor(exmp, self.max_len + 2)

        # return src, src_len, trg, trg_input, trg_len, bert_src, bert_trg, bert_exmp, content_trg, content_len
        return src, trg, trg_input, bert_src, bert_trg, bert_exmp, content_trg



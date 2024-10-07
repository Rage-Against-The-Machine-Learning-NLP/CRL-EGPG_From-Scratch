import os, sys
import random
from typing import List
from utils import pkl_load

from torch.utils.data import Dataset, DataLoader
import torch


class OurDataset(Dataset):
    def __init__(self, dataroot: str, max_len: int, type: str):
        def resolve_paths() -> List[str]:
            """
            returns absolute paths of all files associated with the dataset (train/test/valid)
            resolved against the root path provided
            """

            files = (
                "src",
                "src_bert_ids",
                "src_tags",
                "trg",
                "trg_bert_ids",
                "trg_tags",
                "similarity"
            )

            dirname = os.path.dirname(__file__)
            abspaths = []
            for file in files:
                relpath = dataroot + "/" + f"{self.type}_" + file + ".pkl"
                abspath = os.path.join(dirname, relpath)
                sys.path.append(abspath) # TODO: maybe there's a better way to handle this
                abspaths.append(abspath)
            return abspaths
        

        super().__init__()

        assert type in ["train", "test", "valid"]
        self.type = type
        self.max_len: int = max_len

        abspaths = resolve_paths()

        self.src = pkl_load(abspaths[0])
        self.src_bert_ids = pkl_load(abspaths[1])
        self.src_tags = pkl_load(abspaths[2])
        self.trg = pkl_load(abspaths[3])
        self.trg_bert_ids = pkl_load(abspaths[4])
        self.trg_tags = pkl_load(abspaths[5])
        self.similarity = pkl_load(abspaths[6])


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

        return src, content_trg, trg, trg_input, bert_src, bert_trg, bert_exmp


if __name__ == "__main__":
    # some basic code to test if the dataset is working fine

    train = OurDataset("../data/quora/processed", 15, "train")
    val = OurDataset("../data/quora/processed", 15, "valid")
    test = OurDataset("../data/quora/processed", 15, "test")

    train_dl = DataLoader(train, batch_size=64, shuffle=False)
    val_dl = DataLoader(val, batch_size=64, shuffle=False)
    test_dl = DataLoader(test, batch_size=64, shuffle=False)

    # works fine, uncomment if needed
    # count = 0
    # for data in train_dl:
    #     if count >= 10:
    #         break
    #     count += 1
    #     print(count, end="\t")
    #     print([d.shape for d in data])

    # print("*" * 10)
    # count = 0
    # for data in test_dl:
    #     if count >= 10:
    #         break
    #     count += 1
    #     print(count, end="\t")
    #     print([d.shape for d in data])

    # print("*" * 10 )
    # count = 0
    # for data in val_dl:
    #     if count >= 10:
    #         break
    #     count += 1
    #     print(count, end="\t")
    #     print([d.shape for d in data])
    
    # print("*" * 10 )
    
    entry = train_dl.dataset[0]
    for t in entry:
        print(t.shape, t)
    
            
    

from enum import Enum

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Seq2SeqModel(Enum):
    GRU = "gru"
    LSTM = "lstm"


class Seq2SeqEncoder(nn.Module):
    def __init__(
        self,
        model_type: Seq2SeqModel = Seq2SeqModel.GRU,
        hidden_dim: int = 256,
        input_dim: int = 256,
        num_layers: int = 1,
        drop_out: float = 0.2,
        bidirectional: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        args = {
            "input_size": input_dim,
            "hidden_size": hidden_dim,
            "num_layers": num_layers,
            "bias": True,
            "batch_first": True,
            "dropout": 0 if num_layers == 1 else drop_out,
            "bidirectional": bidirectional,
            "device": device,
        }

        match model_type:
            case Seq2SeqModel.GRU:
                self.model = nn.GRU(**args)
            case Seq2SeqModel.LSTM:
                self.model = nn.LSTM(**args)
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")

    def forward(self, seq_arr: torch.Tensor, seq_len: torch.Tensor):
        # NOTE: the original code sorts the input sequences by length, and restores the order.
        # NOTE:     this is no longer necessary, as below:
        # NOTE: https://discuss.pytorch.org/t/why-lengths-should-be-given-in-sorted-order-in-pack-padded-sequence/3540/8

        padded_input = pack_padded_sequence(
            seq_arr, seq_len, batch_first=True, enforce_sorted=False
        )

        # todo: will this still work if num_layers > 1?
        # original code only seems to test for num_layers == 1
        output, hidden = self.model(padded_input)
        output, _ = pad_packed_sequence(output, batch_first=True)
        hidden = torch.cat(tuple(hidden), dim=-1)

        return output, hidden

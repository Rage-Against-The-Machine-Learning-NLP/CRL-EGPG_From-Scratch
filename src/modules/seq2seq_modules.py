from enum import Enum

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .attention import ScaledDotProductAttention


class Seq2SeqModelType(Enum):
    GRU = "gru"
    LSTM = "lstm"


class Seq2SeqEncoder(nn.Module):
    def __init__(
        self,
        model_type: Seq2SeqModelType = Seq2SeqModelType.GRU,
        hidden_dim: int = 256,
        input_dim: int = 256,
        num_layers: int = 1,
        drop_out: float = 0.2,
        bidirectional: bool = True,
        device: torch.device = torch.device(device="cpu"),
    ) -> None:
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
            case Seq2SeqModelType.GRU:
                self.model = nn.GRU(**args)
            case Seq2SeqModelType.LSTM:
                self.model = nn.LSTM(**args)
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")

    def forward(
        self, seq_arr: torch.Tensor, seq_len: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: the original code sorts the input sequences by length, and restores the order.
        # NOTE:     this is no longer necessary, as below:
        # NOTE: https://discuss.pytorch.org/t/why-lengths-should-be-given-in-sorted-order-in-pack-padded-sequence/3540/8

        padded_input = pack_padded_sequence(
            input=seq_arr, lengths=seq_len, batch_first=True, enforce_sorted=False
        )

        # todo: will this still work if num_layers > 1? original configs only have n_l==1
        output, hidden = self.model(padded_input)
        output, _ = pad_packed_sequence(sequence=output, batch_first=True)
        hidden = torch.cat(tensors=tuple(hidden), dim=-1)

        return output, hidden


class Seq2SeqDecoder(nn.Module):
    def __init__(
        self,
        word_emb_layer: nn.Embedding,
        style_attention_input_dim: int,
        model_type: Seq2SeqModelType = Seq2SeqModelType.GRU,
        encoder_final_out_dim: int = 256,
        decoder_hidden_dim: int = 256,
        vocabulary_dim: int = 1,
        num_layers: int = 1,
        drop_out: float = 0.2,
        input_dim: int = 256,
        hidden_dim: int = 256,
        device: torch.device = torch.device(device="cpu"),
    ):

        super().__init__()

        self.device = device
        self.W_enc2dec = nn.Linear(
            encoder_final_out_dim + style_attention_input_dim,
            decoder_hidden_dim,
            bias=True,
        )
        self.word_emb_layer = word_emb_layer
        self.attention_layer = ScaledDotProductAttention(
            decoder_hidden_dim, encoder_final_out_dim, device
        )

        args = {
            "input_size": input_dim,
            "hidden_size": hidden_dim,
            "num_layers": num_layers,
            "bias": True,
            "batch_first": True,
            "dropout": 0 if num_layers == 1 else drop_out,
            "bidirectional": False,
            "device": device,
        }
        match model_type:
            case Seq2SeqModelType.GRU:
                self.model = nn.GRU(**args)
            case Seq2SeqModelType.LSTM:
                self.model = nn.LSTM(**args)
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")

        self.projection_layer = nn.Linear(hidden_dim, vocabulary_dim)
        self.mode = "train"

    def set_mode(self, mode: str):
        if mode not in ["train", "eval", "infer"]:
            raise ValueError(f"Unknown mode: {self.mode}")
        self.mode = mode

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_input: torch.Tensor,
        style_emb: torch.Tensor,
        max_seq_len: int = 21,
    ) -> torch.Tensor:

        style_feature = style_emb[:, -1, :]
        encoder_hidden = torch.concat([encoder_hidden, style_feature], dim=-1)
        hidden: torch.Tensor = self.W_enc2dec(encoder_hidden).unsqueeze(0)

        return_val: torch.Tensor # cheap hack for neatness

        if self.mode in ["train", "eval"]:
            decoder_input_emb = self.word_emb_layer(decoder_input)
            decoder_output_arr = []

            for step in range(decoder_input.size()[-1]):
                context, _ = self.attention_layer(
                    hidden[-1], encoder_output, encoder_output, encoder_mask
                )
                output, hidden = self.model(
                    torch.concat(
                        [context, decoder_input_emb[:, step]], dim=-1
                    ).unsqueeze(dim=1),
                    hidden,
                )
                decoder_output_arr.append(output.squeeze(dim=1))

            decoder_output = self.projection_layer(
                torch.stack(decoder_output_arr, dim=1)
            )
            return_val = decoder_output

        elif self.mode == "infer":
            id_arr = []
            previous_vec = self.word_emb_layer(
                torch.ones(
                    size=[encoder_output.size()[0]],
                    dtype=torch.long,
                    device=self.device,
                )
                * torch.tensor(2, dtype=torch.long, device=self.device)
            )

            for step in range(max_seq_len):
                context, _ = self.attention_layer(
                    hidden[-1], encoder_output, encoder_output, encoder_mask
                )
                output, hidden = self.model(
                    torch.concat([context, previous_vec], dim=-1).unsqueeze(dim=1),
                    hidden,
                )

                decoder_output = self.projection_layer(output.squeeze(dim=1))
                _, previous_id = decoder_output.max(dim=-1, keepdim=False)
                previous_vec = self.word_emb_layer(previous_id)
                id_arr.append(previous_id)

            decoded_ids = torch.stack(id_arr, dim=1)
            return_val = decoded_ids
        
        return return_val

import torch
import torch.nn as nn

from src.parse_config import ModelConfig
from .seq2seq_modules import Seq2SeqEncoder, Seq2SeqDecoder, Seq2SeqModelType

import src.utils as utils

class Seq2Seq(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        model_type: Seq2SeqModelType = Seq2SeqModelType.GRU,
        device: torch.device = torch.device(device="cpu"),
    ):

        super().__init__()
        self.emb_layer = nn.Embedding(
            num_embeddings=config.vocabulary_dim,
            embedding_dim=config.embedding_dim,
            padding_idx=0,
        )

        glove_weight = utils.initialise_word_embedding(config.glove_file, config.vocab_file)
        self.emb_layer.weight.data.copy_(torch.from_numpy(glove_weight))
        self.device = device

        self.encoder_layer = Seq2SeqEncoder(
            model_type=model_type,
            hidden_dim=config.encoder.hidden_dim,
            input_dim=config.encoder.input_dim,
            num_layers=config.encoder.num_layers,
            drop_out=config.encoder.drop_out,
            bidirectional=config.encoder.bidirectional,
            device=self.device,
        )
        self.decoder = Seq2SeqDecoder(
            word_emb_layer=self.emb_layer,
            style_attention_input_dim=config.style_attn.style_in,
            model_type=model_type,
            encoder_final_out_dim=config.encoder.final_out_dim,
            decoder_hidden_dim=config.encoder.hidden_dim,  # TODO: why?
            vocabulary_dim=config.vocabulary_dim,
            num_layers=config.decoder.num_layers,
            drop_out=config.decoder.drop_out,
            input_dim=config.decoder.input_dim,
            hidden_dim=config.decoder.hidden_dim,
            device=self.device,
        )

        self.params = list(filter(lambda x: x.requires_grad, self.parameters()))
        self.opttimizer = torch.optim.Adam(params=self.params, lr=config.learning_rate)

    def forward(
        self,
        seq_arr: torch.Tensor,
        seq_len: torch.Tensor,
        style_emb: torch.Tensor,
        response: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        max_seq_len: int = 16,
        mode: str = "train"
    ):
        
        seq_arr = seq_arr.to(self.device)
        seq_len = seq_len.to(self.device)
        style_emb = style_emb.to(self.device)
        if response is not None:
            response = response.to(self.device)
        if decoder_input is not None:
            decoder_input = decoder_input.to(self.device)

        encoder_mask = (seq_arr == 0).byte()
        seq_arr = self.emb_layer(seq_arr)
        encoder_output, encoder_hidden = self.encoder_layer(seq_arr, seq_len)
        output = self.decoder(
            encoder_hidden,
            encoder_output,
            encoder_mask,
            response,
            decoder_input,
            style_emb,
            max_seq_len=max_seq_len,
            mode=mode
        )
        return output, encoder_hidden


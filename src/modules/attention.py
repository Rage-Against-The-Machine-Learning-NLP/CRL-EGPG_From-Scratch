import torch
import torch.nn as nn

import numpy as np


# used for cross attention
class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        decoder_hidden_dim: int = 256,
        encoder_final_out_dim: int = 256,
        device: torch.device = torch.device(device="cpu"),
    ):
        super().__init__()

        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(
                [
                    decoder_hidden_dim,  # queries' dim
                    encoder_final_out_dim,  # keys' dim
                ],
                device=device,
                requires_grad=True,
            ),
            requires_grad=True,
        )

        nn.init.normal_(
            self.weight,
            mean=0,
            std=np.sqrt(2 / (decoder_hidden_dim + encoder_final_out_dim)),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        attn_weight: torch.Tensor = key.bmm(
            query.mm(self.weight).unsqueeze(dim=2)
        ).squeeze(dim=2)
        if bias is not None:
            attn_weight += bias

        # Convert mask to boolean type
        mask = mask[:, : attn_weight.shape[-1]].bool()
        attn_weight.masked_fill_(mask, -np.inf)
        attn_weight = attn_weight.softmax(dim=-1)
        attn_out: torch.Tensor = (attn_weight.unsqueeze(dim=2) * value).sum(dim=1)
        return attn_out, attn_weight

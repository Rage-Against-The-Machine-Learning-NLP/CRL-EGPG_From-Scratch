import torch
from torch import nn
import torch.nn.functional as F


def get_nll_loss(
    fc_out: torch.Tensor, fc_label: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    :param fc_out:      [b, v]  [b, s, v]
    :param fc_label:    [b]     [b, s]
    :param reduction:   {'none', 'mean'}
    :return:            'mean': []      'none': [b]
    """

    min_num = 1e-6
    fc_out = F.log_softmax(fc_out, dim=-1)
    label: torch.Tensor = (fc_label > 0).float()

    return_val: torch.Tensor  # cheap hack for neatness

    if fc_label.ndim == 1:
        return_val = label * F.nll_loss(
            input=fc_out, target=fc_label, reduction=reduction
        )
    elif fc_label.ndim == 2:
        loss = (
            label
            * F.nll_loss(
                input=fc_out.transpose(1, 2), target=fc_label, reduction="none"
            )
        ).sum(dim=-1) / (label.sum(dim=-1) + min_num)

        return_val = loss.mean() if reduction == "mean" else loss
    else:
        raise ValueError(
            f"Expected one or two number of dims for `fc_label`, got {fc_label.ndim}"
        )

    return return_val


class UnsupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    Only acting as unsupervised contrastive loss in SimCLR"""

    def __init__(
        self,
        temperature: float = 0.5,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
        device: torch.device = torch.device(device="cpu"),
    ):
        super().__init__()
        self.temperature: float = temperature
        self.contrast_mode: str = contrast_mode
        self.base_temperature: float = base_temperature
        self.device = device

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute loss for model, degenerated to SimCLR
        unsupervised loss: https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
        Returns:
            A loss scalar.
        """

        if features.ndim < 3:
            raise ValueError(">= 3 dimensions required")
        if features.ndim > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32, device=self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature, anchor_count = contrast_feature, contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0,
        )
        mask *= logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss: torch.Tensor = (
            -(self.temperature / self.base_temperature) * mean_log_prob_pos
        )
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

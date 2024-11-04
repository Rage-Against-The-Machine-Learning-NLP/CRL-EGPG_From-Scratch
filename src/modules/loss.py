import torch
import torch.nn.functional as F


def get_nll_loss(fc_out: torch.Tensor, fc_label: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    :param fc_out:      [b, v]  [b, s, v]
    :param fc_label:    [b]     [b, s]
    :param reduction:   {'none', 'mean'}
    :return:            'mean': []      'none': [b]
    """

    min_num = 1e-6
    fc_out = F.log_softmax(fc_out, dim=-1)
    label: torch.Tensor = (fc_label > 0).float()

    if fc_label.ndim == 1:
        return label * F.nll_loss(input=fc_out, target=fc_label, reduction=reduction)
    elif fc_label.ndim == 2:
        loss = (
            label
            * F.nll_loss(
                input=fc_out.transpose(1, 2), target=fc_label, reduction="none"
            )
        ).sum(dim=-1) / (label.sum(dim=-1) + min_num)

        return loss.mean() if reduction == "mean" else loss
    return None # TODO: or raise error?


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from src.modules.seq2seq import Seq2Seq
from src.modules.style import StyleExtractor
from src.modules.loss import get_nll_loss, UnsupervisedContrastiveLoss


def train_loop(
    epoch: int,
    lambda_1: float,
    lambda_2: float,
    seq2seq: Seq2Seq,
    style_extractor: StyleExtractor,
    optimizer: Optimizer,
    train_dl: DataLoader,
    device: torch.device = torch.device(device="cpu"),
) -> list[float]:

    criterion = UnsupervisedContrastiveLoss(device=device)
    seq2seq.train()
    seq2seq.set_decoder_mode("train")
    style_extractor.train()

    base_losses = []
    content_losses = []
    style_losses = []

    for (
        src,
        src_len,
        content_trg,
        content_len,
        trg,
        trg_input,
        bert_src,
        bert_trg,
        bert_exmp,
    ) in tqdm(train_dl, desc=f"Training epoch {epoch}"):

        src: torch.Tensor
        src_len: torch.Tensor
        content_trg: torch.Tensor
        content_len: torch.Tensor
        trg: torch.Tensor
        trg_input: torch.Tensor
        bert_src: torch.Tensor
        bert_trg: torch.Tensor
        bert_exmp: torch.Tensor

        optimizer.zero_grad()

        (
            src,
            src_len,
            content_trg,
            content_len,
            trg,
            trg_input,
            bert_src,
            bert_trg,
            bert_exmp,
        ) = (
            src.to(device),
            src_len.to(device),
            content_trg.to(device),
            content_len.to(device),
            trg.to(device),
            trg_input.to(device),
            bert_src.to(device),
            bert_trg.to(device),
            bert_exmp.to(device),
        )

        # base optimization objective
        trg_style_emb: torch.Tensor = style_extractor(bert_trg)
        exmp_style_emb: torch.Tensor = style_extractor(bert_exmp)
        predicted_trg, src_hidden = seq2seq.forward(
            src, src_len, exmp_style_emb, decoder_input=trg_input
        )
        pred_trg_loss: torch.Tensor = get_nll_loss(predicted_trg, trg, reduction="mean")

        # content contrast calculation
        content_trg_emb = seq2seq.emb_layer(content_trg)
        _, trg_hidden = seq2seq.encoder_layer(content_trg_emb, content_len)
        content_src = torch.unsqueeze(F.normalize(src_hidden, dim=1), 1)
        content_trg = torch.unsqueeze(F.normalize(trg_hidden, dim=1), 1)
        content_contrast: torch.Tensor = torch.concat((content_src, content_trg), dim=1)

        # style contrast calculation
        style_trg = trg_style_emb[:, -1, :]
        style_exmp = exmp_style_emb[:, -1, :]
        style_trg = torch.unsqueeze(F.normalize(style_trg, dim=1), 1)
        style_exmp = torch.unsqueeze(F.normalize(style_exmp, dim=1), 1)
        style_contrast: torch.Tensor = torch.cat((style_trg, style_exmp), dim=1)

        content_loss: torch.Tensor = criterion(content_contrast)  # CCL
        style_loss: torch.Tensor = criterion(style_contrast)  # SCL

        total_nll_loss = pred_trg_loss + lambda_1 * content_loss + lambda_2 * style_loss
        total_nll_loss.backward()
        optimizer.step()

        base_losses.append(pred_trg_loss.cpu().item())
        content_losses.append(content_loss.cpu().item())
        style_losses.append(style_loss.cpu().item())

    avg_base_loss: float = float(np.mean(base_losses))
    avg_content_loss: float = float(np.mean(content_losses))
    avg_style_loss: float = float(np.mean(style_losses))
    print(
        f"Epoch {epoch} losses: base: {avg_base_loss}, content: {avg_content_loss}, style: {avg_style_loss}"
    )
    return [avg_base_loss, avg_content_loss, avg_style_loss]


def eval_loop(
    seq2seq: Seq2Seq,
    style_extractor: StyleExtractor,
    eval_dl: DataLoader,
    epoch: int,
    device: torch.device = torch.device(device="cpu"),
):
    # Uses only traditional evaluation loss, not contrastive loss (only used for training)

    seq2seq.eval()
    seq2seq.set_decoder_mode("eval")
    style_extractor.eval()

    perplexities = []
    nll_losses = []

    with torch.no_grad():
        for (
            src,
            src_len,
            content_trg,
            content_len,
            trg,
            trg_input,
            bert_src,
            bert_trg,
            bert_exmp,
        ) in tqdm(eval_dl, desc=f"Evaluating epoch {epoch}"):

            src: torch.Tensor
            src_len: torch.Tensor
            content_trg: torch.Tensor
            content_len: torch.Tensor
            trg: torch.Tensor
            trg_input: torch.Tensor
            bert_src: torch.Tensor
            bert_trg: torch.Tensor
            bert_exmp: torch.Tensor

            (
                src,
                src_len,
                content_trg,
                content_len,
                trg,
                trg_input,
                bert_src,
                bert_trg,
                bert_exmp,
            ) = (
                src.to(device),
                src_len.to(device),
                content_trg.to(device),
                content_len.to(device),
                trg.to(device),
                trg_input.to(device),
                bert_src.to(device),
                bert_trg.to(device),
                bert_exmp.to(device),
            )

            exmp_style_emb = style_extractor(bert_exmp)
            predicted_trg, _ = seq2seq.forward(
                src, src_len, exmp_style_emb, decoder_input=trg_input
            )

            pred_trg_loss: torch.Tensor = get_nll_loss(
                predicted_trg, trg, reduction="none"
            )
            perplexity, nll_loss = pred_trg_loss.exp(), pred_trg_loss.mean()
            perplexities.append(perplexity)
            nll_losses.append(nll_loss.cpu().item())

        ppl = torch.cat(perplexities, dim=0)
        # filter out perplexity values > 200
        ppl_mask = (ppl < 200).float()  # overloading BS
        # get average of values < 200
        ppl = (ppl * ppl_mask).sum() / ppl_mask.sum()

        avg_ppl: float = ppl.cpu().item()
        avg_nll: float = float(np.mean(nll_losses))

        print(f"Evaluation averages: NLL loss: {avg_nll}, Perplexity: {avg_ppl}")

    return [avg_nll, avg_ppl]

import os
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

import src.utils as utils
from src.parse_config import load_config_from_json, ModelConfig
from src.dataset import get_dataloaders
from src.modules.seq2seq import Seq2Seq
from src.modules.seq2seq_modules import Seq2SeqModelType, get_seq2seqmodeltype
from src.modules.style import StyleExtractor
from src.modules.loops import train_loop, eval_loop


def train_model(
    num_epochs: int,
    lambda_1: float,
    lambda_2: float,
    temperature: float,
    base_temperature: float,
    seq2seq: Seq2Seq,
    style_extractor: StyleExtractor,
    optimizer: Optimizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    model_save_dir: str,
    train_losses_file: str,
    validation_losses_file: str,
    device: torch.device = torch.device(device="cpu"),
):
    """Trains the model and saves losses for all epochs in the given files"""

    train_loss_arr = []
    eval_loss_arr = []
    seq2seq.to(device)
    style_extractor.to(device)
    for epoch in range(num_epochs):
        # base, content, style losses
        train_losses: list[float] = train_loop(
            epoch,
            lambda_1,
            lambda_2,
            temperature,
            base_temperature,
            seq2seq,
            style_extractor,
            optimizer,
            train_dl,
            device,
        )
        # averaged nll loss, masked perplexity, unmasked perplexity
        eval_losses: list[float] = eval_loop(
            seq2seq, style_extractor, val_dl, epoch, device
        )
        train_loss_arr.append(train_losses)
        eval_loss_arr.append(eval_losses)

    # save last epoch's model only
    torch.save(
        seq2seq.state_dict(),
        os.path.join(model_save_dir, "seq2seq.pkl"),
    )
    torch.save(
        style_extractor.state_dict(),
        os.path.join(model_save_dir, "style_extractor.pkl"),
    )

    utils.pkl_dump(train_loss_arr, train_losses_file)
    utils.pkl_dump(eval_loss_arr, validation_losses_file)


def main(config_file: str):
    config: ModelConfig = load_config_from_json(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    print(f"encoder model type: {config.encoder.model_type}")
    print(f"decoder model type: {config.decoder.model_type}")
    print(f"learning rate: {config.training.learning_rate}")
    print(f"lambda1: {config.training.lambda_1}")
    print(f"lambda2: {config.training.lambda_2}")
    print(f"temperature: {config.training.temperature}")
    print(f"base temperature: {config.training.base_temperature}")

    train_dl, val_dl, _ = get_dataloaders(
        config.dataset_dir,
        config.training.max_sent_len,
        config.training.batch_size,
    )

    seq2seq = Seq2Seq(config, device)
    style_extractor = StyleExtractor(config.training.style_extractor_model_type)
    params: list[nn.parameter.Parameter] = list(seq2seq.parameters()) + list(
        style_extractor.parameters()
    )
    optimizer = Adam(params, lr=config.training.learning_rate)

    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)

    train_model(
        config.training.num_epochs,
        config.training.lambda_1,
        config.training.lambda_2,
        config.training.temperature,
        config.training.base_temperature,
        seq2seq,
        style_extractor,
        optimizer,
        train_dl,
        val_dl,
        config.model_save_dir,
        config.training.train_losses_file,
        config.training.validation_losses_file,
        device,
    )


if __name__ == "__main__":
    CONFIG_FILE = utils.resolve_relpath("./config.json")
    main(CONFIG_FILE)

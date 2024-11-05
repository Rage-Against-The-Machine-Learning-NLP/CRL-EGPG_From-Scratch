import os
import json
from dataclasses import dataclass

from .utils import resolve_relpath


@dataclass
class EncoderConfig:
    hidden_dim: int
    input_dim: int
    num_layers: int
    bidirectional: bool
    final_out_dim: int
    drop_out: int


@dataclass
class DecoderConfig:
    hidden_dim: int
    input_dim: int
    num_layers: int
    drop_out: int


@dataclass
class StyleAttentionConfig:
    style_in: int
    style_out: int


@dataclass
class TrainingConfig:
    save_freq: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    max_sent_len: int
    seq2seq_model_type: str
    style_extractor_model_type: str
    model_save_dir: str
    loss_file: str
    perplexity_file: str


@dataclass
class ModelConfig:
    model_name: str
    glove_file: str
    dataset_dir: str
    vocab_file: str
    embedding_dim: int
    vocabulary_dim: int
    encoder: EncoderConfig
    decoder: DecoderConfig
    style_attn: StyleAttentionConfig
    training: TrainingConfig


def load_config_from_json(file_path: str) -> ModelConfig:
    with open(file_path, "r") as f:
        data = json.load(f)

    # Convert nested dictionaries into dataclass instances
    encoder_config = EncoderConfig(**data["encoder"])
    decoder_config = DecoderConfig(**data["decoder"])
    style_attn_config = StyleAttentionConfig(**data["style_attn"])
    training_config = TrainingConfig(**data["training"])

    # resolve paths
    training_config.model_save_dir = resolve_relpath(training_config.model_save_dir)
    training_config.loss_file = resolve_relpath(training_config.loss_file)
    training_config.perplexity_file = resolve_relpath(training_config.perplexity_file)

    # Create the main ModelConfig instance
    model_config = ModelConfig(
        model_name=data["model_name"],
        glove_file=resolve_relpath(
            data["glove_file"]
        ),
        dataset_dir=resolve_relpath(data["dataset_dir"]),
        vocab_file=resolve_relpath(
            os.path.join(data["dataset_dir"], data["vocab_file"])
        ),
        embedding_dim=data["embedding_dim"],
        vocabulary_dim=data["vocabulary_dim"],
        encoder=encoder_config,
        decoder=decoder_config,
        style_attn=style_attn_config,
        training=training_config,
    )

    return model_config


if __name__ == "__main__":
    model_config = load_config_from_json(resolve_relpath("../config.json"))
    print(model_config)

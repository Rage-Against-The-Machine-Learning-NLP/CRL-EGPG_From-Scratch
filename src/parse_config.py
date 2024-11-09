import os
import json
from dataclasses import dataclass

from .utils import resolve_relpath


@dataclass
class EncoderConfig:
    model_type: str
    hidden_dim: int
    input_dim: int
    num_layers: int
    bidirectional: bool
    final_out_dim: int
    drop_out: int


@dataclass
class DecoderConfig:
    model_type: str
    hidden_dim: int
    input_dim: int
    num_layers: int
    drop_out: int
    bidirectional: bool


@dataclass
class StyleAttentionConfig:
    style_in: int
    style_out: int


@dataclass
class TrainingConfig:
    num_epochs: int
    max_sent_len: int
    batch_size: int
    lambda_1: float
    lambda_2: float
    learning_rate: float
    temperature: float
    base_temperature: float
    style_extractor_model_type: str
    train_losses_file: str
    validation_losses_file: str


@dataclass
class ModelConfig:
    model_name: str
    embedding_dim: int
    vocabulary_dim: int
    glove_file: str
    dataset_dir: str
    vocab_file: str
    model_save_dir: str
    results_dir: str
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
    training_config.train_losses_file = resolve_relpath(
        os.path.join(data["results_dir"], training_config.train_losses_file)
    )
    training_config.validation_losses_file = resolve_relpath(
        os.path.join(data["results_dir"], training_config.validation_losses_file)
    )

    # Create the main ModelConfig instance
    model_config = ModelConfig(
        model_name=data["model_name"],
        embedding_dim=data["embedding_dim"],
        vocabulary_dim=data["vocabulary_dim"],
        glove_file=resolve_relpath(data["glove_file"]),
        dataset_dir=resolve_relpath(data["dataset_dir"]),
        vocab_file=resolve_relpath(
            os.path.join(data["dataset_dir"], data["vocab_file"])
        ),
        model_save_dir=resolve_relpath(data["model_save_dir"]),
        results_dir=resolve_relpath(data["results_dir"]),
        encoder=encoder_config,
        decoder=decoder_config,
        style_attn=style_attn_config,
        training=training_config,
    )

    return model_config


if __name__ == "__main__":
    model_config = load_config_from_json(resolve_relpath("./config.json"))
    print(model_config)

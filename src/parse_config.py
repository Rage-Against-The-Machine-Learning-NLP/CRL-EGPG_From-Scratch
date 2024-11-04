import json
from dataclasses import dataclass

import utils


@dataclass
class EncoderConfig:
    hidden_dim: int
    input_dim: int
    num_layers: int
    bidirectional: int
    final_out_dim: int

@dataclass
class DecoderConfig:
    hidden_dim: int
    input_dim: int

@dataclass
class StyleAttentionConfig:
    style_in: int
    style_out: int

@dataclass
class ModelConfig:
    model_name: str
    embedding_dim: int
    vocabulary_dim: int
    encoder: EncoderConfig
    decoder: DecoderConfig
    style_attn: StyleAttentionConfig


def load_config_from_json(file_path: str) -> ModelConfig:
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    # Convert nested dictionaries into dataclass instances
    encoder_config = EncoderConfig(**data['encoder'])
    decoder_config = DecoderConfig(**data['decoder'])
    style_attn_config = StyleAttentionConfig(**data['style_attn'])
    
    # Create the main ModelConfig instance
    model_config = ModelConfig(
        model_name=data['model_name'],
        embedding_dim=data['embedding_dim'],
        vocabulary_dim=data['vocabulary_dim'],
        encoder=encoder_config,
        decoder=decoder_config,
        style_attn=style_attn_config
    )
    
    return model_config


if __name__ == "__main__":
    model_config = load_config_from_json(utils.resolve_relpath("../config.json"))
    print(model_config)

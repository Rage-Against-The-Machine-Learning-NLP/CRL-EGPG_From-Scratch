# Contrastive Representation Learning for Exemplar-Guided Paraphrase Generation

A from-scratch PyTorch implementation of "Contrastive Representation Learning for Exemplar-Guided Paraphrase Generation" with comprehensive ablation studies and architectural improvements.

## Overview

This project implements a novel approach to paraphrase generation that uses exemplar sentences to guide the style while preserving the semantic content of the source sentence. The model employs contrastive learning to disentangle content and style representations, enabling controllable text generation with arbitrary style exemplars.

### Key Features
- From-scratch implementation in PyTorch with clean, modular architecture
- Comprehensive ablation studies on encoders, decoders, and transformer models
- Extensive hyperparameter analysis for contrastive loss weights
- Model compression with quantization support
- Multiple dataset support (QQP-Pos, ParaNMT)
- Flexible architecture supporting RNN/LSTM/GRU and BERT/RoBERTa/ALBERT variants

## Architecture

The system consists of three main components:

1. **Content Encoder**: Extracts semantic information from source sentences
2. **Style Extractor**: Captures stylistic features from exemplar sentences using transformer models
3. **Decoder**: Generates paraphrases by combining content and style representations

The model is trained using three loss functions:
- **Generation Loss**: Standard negative log-likelihood
- **Content Contrastive Loss (CCL)**: Ensures content preservation
- **Style Contrastive Loss (SCL)**: Enables style transfer

## Results

Our implementation achieves results comparable to the original paper:

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR |
|-------|------|---------|---------|---------|---------|
| Original Paper | 45.40 | 70.52 | 52.06 | 72.73 | 45.10 |
| Our Implementation | 44.73 | 69.99 | 51.45 | 72.47 | 44.28 |

### Best Performing Variants
- **RoBERTa Style Encoder**: BLEU 46.68 (+4.3% improvement)
- **ALBERT Style Encoder**: BLEU 46.53 (+4.0% improvement)
- **Optimal λ=0.3**: BLEU 47.34 (+5.8% improvement)

## Installation

```bash
git clone https://github.com/your-username/CRL-EGPG.git
cd CRL-EGPG

conda env create -f envs.yml
conda activate crl-egpg

pip install editdistance nltk transformers
```

## Project Structure

```
CRL-EGPG/
├── src/
│   ├── modules/
│   │   ├── seq2seq.py          # Main sequence-to-sequence model
│   │   ├── seq2seq_modules.py  # Encoder/decoder components
│   │   ├── style.py            # BERT-based style extractor
│   │   ├── attention.py        # Attention mechanisms
│   │   ├── loss.py             # Contrastive loss functions
│   │   └── loops.py            # Training/evaluation loops
│   ├── dataset.py              # Data loading and processing
│   ├── process.py              # Dataset preprocessing
│   ├── train.py                # Training script
│   ├── generate.py             # Text generation
│   ├── prep-eval.py            # Evaluation preparation
│   └── config.json             # Model configuration
├── data/                       # Dataset directory
├── models/                     # Saved model checkpoints
├── results/                    # Evaluation outputs
└── glove/                      # GloVe embeddings
```

## Usage

### Data Preprocessing

```bash
# Process QQP-Pos dataset
python -m src.process --dataset quora

# Process ParaNMT dataset
python -m src.process --dataset para
```

### Training

#### Basic Training
```bash
python -m src.train
```

#### Custom Configuration
Modify `src/config.json` to experiment with different architectures:

```json
{
  "training": {
    "style_extractor_model_type": "roberta",
    "lambda_1": 0.3,
    "lambda_2": 0.3
  },
  "encoder": {
    "model_type": "lstm"
  },
  "decoder": {
    "model_type": "gru"
  }
}
```

### Generation and Evaluation

```bash
# Generate paraphrases
python -m src.generate

# Prepare evaluation data
python -m src.prep-eval
```

## Ablation Studies

### Style Extractor Variants
- BERT (baseline)
- RoBERTa (+4.3% BLEU improvement)
- ALBERT (+4.0% BLEU improvement)

### Architecture Variants
Tested combinations of RNN/LSTM/GRU for encoder-decoder pairs:
- GRU-LSTM: BLEU 44.88
- LSTM-GRU: BLEU 45.23 (best alternative)
- RNN variants: BLEU 42.04-42.59

### Hyperparameter Analysis
Systematic testing of contrastive loss weights (λ₁, λ₂) from 0.001 to 2.5:
- Optimal range: 0.3-0.7
- Peak performance: λ=0.3 (BLEU 47.34)
- Both losses essential: reducing either to 0.01 significantly hurts performance

## Key Findings

1. **Style Encoder Impact**: RoBERTa and ALBERT significantly outperform BERT
2. **Architecture Robustness**: Framework works well across different RNN architectures
3. **Hyperparameter Sensitivity**: Contrastive loss weights are crucial (optimal: 0.3-0.7)
4. **Model Compression**: 54% size reduction via quantization with minimal performance loss
5. **Loss Balance**: Both content and style losses are essential

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- NLTK
- NumPy
- editdistance

## Datasets

### QQP-Pos
- Source: Quora Question Pairs
- Size: 137k train, 3k test, 3k validation
- Formal question pairs marked as duplicates

### ParaNMT
- Source: Back-translated paraphrase pairs  
- Size: 493k train, 800 test, 500 validation
- Automatically generated paraphrases

## Reference

This work is an ablation study based on the paper:
```
Contrastive Representation Learning for Exemplar-Guided Paraphrase Generation
Haoran Yang, Wai Lam, Piji Li
EMNLP 2021 Findings
```

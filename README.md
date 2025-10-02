# Cell2Gene: Spatial Gene Expression Prediction

A modular implementation for predicting spatial gene expression from histology images using Graph Neural Networks and Transformers.

## Project Structure

```
Cell2Gene/
├── models/
│   ├── __init__.py
│   ├── gnn.py              # Graph Neural Network models (GAT, GCN)
│   └── transformer.py     # Transformer-based predictor
├── dataset.py              # HEST dataset handling
├── trainer.py              # Training functions
├── utils.py                # Utility functions and evaluation metrics
├── train.py                # Main training script
├── logs/                   # Training logs and results
├── checkpoints/            # Model checkpoints
└── data/                   # Data storage
```

## Usage

### Quick Start

```bash
cd /data/yujk/hovernet2feature/Cell2Gene
python train.py
```

### Key Features

- **Modular Design**: Clean separation of models, data, training, and utilities
- **Multiple GNN Support**: GAT and GCN implementations
- **Transformer Integration**: Spatial-aware transformer for gene expression prediction
- **Cross-Validation**: Built-in 10-fold cross-validation
- **Early Stopping**: Prevent overfitting with configurable early stopping
- **Comprehensive Evaluation**: Multiple metrics including correlations and MSE

### Model Architecture

1. **Graph Neural Network**: Processes cell-level features with spatial relationships
2. **Feature Projection**: Maps GNN outputs to transformer dimensions
3. **Spatial Positional Encoding**: HIST2ST-style position encoding
4. **Transformer Encoder**: Captures long-range spatial dependencies
5. **MLP Output**: Predicts gene expression with Softplus activation

### Configuration

Key parameters in `train.py`:
- `batch_size`: Training batch size (default: 16)
- `learning_rate`: AdamW learning rate (default: 3e-6)
- `num_epochs`: Maximum training epochs (default: 60)
- `patience`: Early stopping patience (default: 50)
- `embed_dim`: Transformer embedding dimension (default: 256)

## Requirements

- PyTorch
- PyTorch Geometric
- scanpy
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- tqdm
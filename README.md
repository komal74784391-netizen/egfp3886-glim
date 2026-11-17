# GLIM: Gated Latent Interaction Modules for Tabular Retrieval

Reference implementation for the KDD 2024 paper "Gated Latent Interaction
Modules for Tabular Retrieval" by K. Sung, P. Adler, and H. Rios.

This repository provides the tabular retrieval model and the synthetic retrieval
benchmark used in that paper. It is unrelated to neural ODEs or continuous-time
models.

## Contents

- `src/glim/interaction.py` - gated latent interaction layer
- `src/glim/tabular.py` - tabular feature encoders
- `scripts/train_retrieval.py` - training script
- `configs/kdd24.yaml` - paper configuration
- `results/README.md` - reported retrieval metrics

## Citation

K. Sung, P. Adler, and H. Rios. "Gated Latent Interaction Modules for Tabular
Retrieval." KDD 2024.

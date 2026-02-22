#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== Preprocessing Klein dataset ==="
python scripts/preprocess_klein.py

echo "=== Experiment 1: PCA 50D ==="
python train_RUOT.py --config config/klein_pca50_config.yaml

echo "=== Experiment 2: PCA 30D ==="
python train_RUOT.py --config config/klein_pca30_config.yaml

echo "=== Experiment 3: DM 5D ==="
python train_RUOT.py --config config/klein_dm5_config.yaml

echo "=== Experiment 4: DM 10D ==="
python train_RUOT.py --config config/klein_dm10_config.yaml

echo "=== All experiments complete ==="

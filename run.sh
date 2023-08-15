#!/usr/bin/zsh

python -m enhancer -c experiments/discriminator/conv.yaml train
python -m enhancer -c experiments/discriminator/res.yaml train
python -m enhancer -c experiments/discriminator/dense.yaml train
python -m enhancer -c experiments/enhancer/conv.yaml train
python -m enhancer -c experiments/enhancer/res.yaml train
python -m enhancer -c experiments/enhancer/dense.yaml train
python -m enhancer -c experiments/enhancer/dense_deep.yaml train

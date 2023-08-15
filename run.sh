#!/usr/bin/zsh

python -m enhancer -c experiments/discriminator/conv.yaml train
python -m enhancer -c experiments/enhancer/conv.yaml train

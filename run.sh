#!/usr/bin/zsh

python -m enhancer -c experiments/conv_dis/config.yaml train
python -m enhancer -c experiments/res_dis/config.yaml train
python -m enhancer -c experiments/dense_dis/config.yaml train
python -m enhancer -c experiments/dense_dis/config_shallow.yaml train
python -m enhancer -c experiments/conv_gen/config.yaml train
python -m enhancer -c experiments/res_gen/config.yaml train
python -m enhancer -c experiments/dense_gen/config.yaml train
python -m enhancer -c experiments/dense_gen/config_shallow.yaml train

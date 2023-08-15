#!/usr/bin/zsh

python -m enhancer -c experiments/conv_dis/config.yaml train
python -m enhancer -c experiments/conv_dis/config1.yaml train
python -m enhancer -c experiments/conv_dis/config2.yaml train
python -m enhancer -c experiments/conv_dis/config3.yaml train
python -m enhancer -c experiments/conv_dis/config4.yaml train
python -m enhancer -c experiments/conv_dis/config5.yaml train
python -m enhancer -c experiments/conv_dis/config6.yaml train
python -m enhancer -c experiments/conv_dis/config7.yaml train
python -m enhancer -c experiments/conv_dis/config8.yaml train
python -m enhancer -c experiments/conv_dis/config9.yaml train
# python -m enhancer -c experiments/res_dis/config.yaml train
# python -m enhancer -c experiments/dense_dis/config.yaml train
# python -m enhancer -c experiments/dense_dis/config_shallow.yaml train
# python -m enhancer -c experiments/conv_gen/config.yaml train
# python -m enhancer -c experiments/res_gen/config.yaml train
# python -m enhancer -c experiments/dense_gen/config.yaml train
# python -m enhancer -c experiments/dense_gen/config_shallow.yaml train

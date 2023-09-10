#!/usr/bin/zsh

# discriminator
#
# python -m enhancer -c experiments/discriminator/conv.yaml train
# python -m enhancer -c experiments/discriminator/res.yaml train
# python -m enhancer -c experiments/discriminator/dense.yaml train
#
# enhancer
#
# python -m enhancer -c experiments/enhancer/conv.yaml train
# python -m enhancer -c experiments/enhancer/res.yaml train
# python -m enhancer -c experiments/enhancer/dense.yaml train
# python -m enhancer -c experiments/enhancer/dense_deep.yaml train
#
# gan
#
# python -m enhancer -c experiments/gan/dense.yaml train
#
# predict
#
# python -m enhancer -c experiments/enhancer/conv_chunks.yaml predict
# python -m enhancer -c experiments/enhancer/res_chunks.yaml predict
# python -m enhancer -c experiments/enhancer/dense_chunks.yaml predict
# python -m enhancer -c experiments/enhancer/dense_deep_chunks.yaml predict
python -m enhancer -c experiments/enhancer/conv.yaml predict
python -m enhancer -c experiments/enhancer/res.yaml predict
python -m enhancer -c experiments/enhancer/dense.yaml predict
python -m enhancer -c experiments/enhancer/dense_deep.yaml predict

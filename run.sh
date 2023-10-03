#!/usr/bin/zsh

# discriminator
#
# python -m enhancer -c experiments/discriminator/conv.yaml --train
# python -m enhancer -c experiments/discriminator/res.yaml --train
# python -m enhancer -c experiments/discriminator/dense.yaml --train
#
# enhancer
#
# python -m enhancer -c experiments/enhancer/conv.yaml --train --predict
# ./bin/do_enhanced.sh conv
# python -m enhancer -c experiments/enhancer/res.yaml --train --predict
# ./bin/do_enhanced.sh res
python -m enhancer -c experiments/enhancer/dense.yaml --train --predict
./bin/do_enhanced.sh dense
# ./bin/do_enhanced.sh dense_chunks
# python -m enhancer -c experiments/enhancer/dense_deep.yaml --train --predict
# ./bin/do_enhanced.sh dense_deep
#
# gan
#
# python -m enhancer -c experiments/gan/dense.yaml --train

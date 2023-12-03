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
# python -m enhancer -c experiments/enhancer/dense.yaml --train --predict
# ./bin/do_enhanced.sh dense
# python -m enhancer -c experiments/enhancer/dense_chunks.yaml --predict
# ./bin/do_enhanced.sh dense_chunks
# python -m enhancer -c experiments/enhancer/dense_deep.yaml --train --predict
# ./bin/do_enhanced.sh dense_deep
#
# gan
#
# python -m enhancer -c experiments/gan/dense.yaml --train
# python -m enhancer -c experiments/gan/dense_fresh.yaml --train
#
# experiments
# python -m enhancer -c experiments/enh/conv.yaml --train --predict
# ./bin/do_enhanced.sh conv2
# python -m enhancer -c experiments/enh/res.yaml --train --predict
# ./bin/do_enhanced.sh res2
# python -m enhancer -c experiments/enh/dense.yaml --train --predict
# ./bin/do_enhanced.sh dense2
# gan
#
# python -m enhancer -c experiments/gan/res.yaml --train --predict
# python -m enhancer -c experiments/gan/dense_fresh.yaml --train
python -m enhancer -c experiments/gan/dense.yaml --train --predict

#! /bin/bash
#

# Script for encoding dataset

# ffmpeg -pix_fmt yuv420p -s 1920x1080 -i test_data/tractor_1080p25.yuv -pix_fmt yuv420p -s 1920x1080  -i enhanced/tractor_1080p25/AI_QP32_ALF0_DB0_SAO0.yuv -filter_complex "psnr" -f null /dev/null > allout.txt 2>&1

EXPERIMENT_NAME=$1
mkdir -p logs/$EXPERIMENT_NAME

for file in $(ls enhanced/*/*.yuv); do
	ffmpeg \
		-pix_fmt yuv420p \
		-s 1920x1080 \
		-i test_data/tractor_1080p25.yuv -pix_fmt yuv420p \
		-s 1920x1080  \
		-i $file \
		-filter_complex "psnr" \
		-f null /dev/null \
		> logs/$EXPERIMENT_NAME/enhanced_$(basename $file) \
		2>&1
	echo $(basename $file)
done

for file in $(ls test_decoded/*.yuv); do
	ffmpeg \
		-pix_fmt yuv420p \
		-s 1920x1080 \
		-i test_data/tractor_1080p25.yuv -pix_fmt yuv420p \
		-s 1920x1080  \
		-i $file \
		-filter_complex "psnr" \
		-f null /dev/null \
		> logs/$EXPERIMENT_NAME/reference_$(basename $file) \
		2>&1
	echo $(basename $file)
done

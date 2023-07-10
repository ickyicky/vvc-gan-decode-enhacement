#! /bin/bash
#

# Script for encoding dataset

for file in $(ls test_decoded/*.yuv); do
	mv $file temp.yuv
	ffmpeg -pix_fmt yuv420p10le -s 1920x1080 -i temp.yuv -pix_fmt yuv420p $file
	rm temp.yuv
done

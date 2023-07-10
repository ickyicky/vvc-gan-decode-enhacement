#! /bin/bash
#

# Script for encoding dataset

for file in $(ls enhanced/*/*.yuv); do
	mv $file temp.yuv
	ffmpeg -pix_fmt yuv444p -s 1920x1080 -i temp.yuv -pix_fmt yuv420p $file
	rm temp.yuv
done

#! /bin/bash
#

# Script for preparing_daataset

[ -d data ] || { echo "Make sure to fetch data into data folder first";  exit 1; }

for FILE in $(ls data/*.mkv); do
	DESTINATION="$(echo $FILE | cut -d "." -f 1).yuv"
	echo "preparing $FILE"
	ffmpeg -y -i $FILE -c:v rawvideo -pixel_format yuv420p $DESTINATION
	mediainfo -f $FILE > $FILE.info
done

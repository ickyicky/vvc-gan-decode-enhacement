#! /bin/bash
#

# Script for preparing_daataset

[ -d "data" ] || { echo "Make sure to fetch data into data folder first";  exit 1; }

for FILE in $(ls data/*.mkv); do
	DESTINATION="$(echo $FILE | cut -d "." -f 1).yuv"
	RESOLUTION="$(echo $FILE | cut -d "/" -f 2 | cut -d "P" -f 1)"
	END_FRAME=$(
	case "$RESOLUTION" in
		("2160") echo "12" ;;
		("1080") echo "24" ;;
		("720") echo "36" ;;
		("480") echo "48" ;;
		("360") echo "60" ;;
	esac)
	echo "preparing $FILE - $END_FRAME frames"
	ffmpeg -i $FILE -c:v rawvideo -vf select="between(n\,0\,$END_FRAME),setpts=PTS-STARTPTS" -pixel_format yuv420p $DESTINATION
done

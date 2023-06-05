#! /bin/bash
#

# Script for encoding dataset

PROFILE=${1}
QP=${2}
ALF=${3}
DB=${4}
SAO=${5}
FILE="${6}"
ODIR="${7}"

print_usage() {
	echo "USAGE:"
	echo "./bin/encode_data.sh PROFILE QP ALF DB SAO FILE ODIR"
}

[[ -z "$PROFILE" ]] && { print_usage; exit 1; }
[[ -z "$QP" ]] && { print_usage; exit 1; }
[[ -z "$ALF" ]] && { print_usage; exit 1; }
[[ -z "$DB" ]] && { print_usage; exit 1; }
[[ -z "$SAO" ]] && { print_usage; exit 1; }
[[ -z "$FILE" ]] && { print_usage; exit 1; }
[[ -z "$ODIR" ]] && { print_usage; exit 1; }

ENCODER_APP="./vvenc/bin/release-static/vvencFFapp"

[[ -d "data" ]] || { echo "Make sure to fetch data into data folder first"; exit 1; }
[[ -d "${ODIR}" ]] || { echo "Make sure to create ${ODIR} first"; exit 1; }

INFO_FILE="$(echo $FILE | cut -d "." -f 1).*.info"

vval() {
	grep "$1" $INFO_FILE \
		| head --lines=1 | \
		python -c "import sys; print(round(float(sys.stdin.read().split(':')[-1])))"
}

WIDTH=$(vval "Width")
HEIGHT=$(vval "Height")
FRAMERATE=$(vval "Frame rate")
END_FRAME=64

SUFFIX="${PROFILE}_QP${QP}_ALF${ALF}_DB${DB}_SAO${SAO}"
DESTINATION="$(echo $FILE | cut -d "." -f 1 | cut -d "/" -f 2)_${SUFFIX}.vvc"
RECON_FILE="$(echo $FILE | cut -d "." -f 1 | cut -d "/" -f 2)_${SUFFIX}_rec.yuv"
LOG_FILE="$(echo $FILE | cut -d "." -f 1 | cut -d "/" -f 2)_${SUFFIX}.log"
echo "Processing ${DESTINATION}..."

CONFIGFILE="cfg/${PROFILE}_ALF${ALF}_DB${DB}_SAO${SAO}.cfg"

${ENCODER_APP} -c ${CONFIGFILE} \
	--InputFile=$FILE \
	--BitstreamFile=$ODIR/$DESTINATION \
	--ReconFile=$ODIR/$RECON_FILE \
	--FrameRate $FRAMERATE \
	--FramesToBeEncoded $END_FRAME \
	--SourceWidth $WIDTH \
	--SourceHeight $HEIGHT \
	--QP=$QP > $ODIR/$LOG_FILE

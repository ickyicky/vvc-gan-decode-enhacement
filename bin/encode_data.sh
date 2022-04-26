#! /bin/bash
#

# Script for encoding dataset

PROFILE=${1}
QP=${2}
ALF=${3}
DB=${4}
SAO=${5}
FILE="data/${6}"

print_usage() {
	echo "USAGE:"
	echo "./bin/encode_data.sh PROFILE QP ALF DB SAO FILE"
}

[[ -z "$PROFILE" ]] && { print_usage; exit 1; }
[[ -z "$QP" ]] && { print_usage; exit 1; }
[[ -z "$ALF" ]] && { print_usage; exit 1; }
[[ -z "$DB" ]] && { print_usage; exit 1; }
[[ -z "$SAO" ]] && { print_usage; exit 1; }
[[ -z "$FILE" ]] && { print_usage; exit 1; }

ODIR="encoded"
ENCODER_APP="./vvc/bin/EncoderAppStatic"

AI_CF="-c vvc/cfg/encoder_intra_vtm.cfg"
RA_CF="-c vvc/cfg/encoder_randomaccess_vtm.cfg"

YESALF="--ALF=1 --CCALF=1"
NOALF="--ALF=0 --CCALF=0"
YESDB="--DeblockingFilterDisable=0"
NODB="--DeblockingFilterDisable=1"
YESSAO="--SAO=1"
NOSAO="--SAO=0"

ARGS=""

[[ -d "data" ]] || { echo "Make sure to fetch data into data folder first"; exit 1; }
[[ -d "vvc" ]] || { echo "Make sure to fetch vvc and compile it into vvc folder first"; exit 1; }
[[ -d "${ODIR}" ]] || { echo "Make sure to create ${ODIR} first"; exit 1; }

ORIG_FILE="$(echo $FILE | cut -d "." -f 1).mkv"
INFO_FILE="$(echo $FILE | cut -d "." -f 1).mkv.info"

vval() {
	grep "$1" $INFO_FILE \
		| head --lines=1 | \
		python -c "import sys; print(round(float(sys.stdin.read().split(':')[-1])))"
}

WIDTH=$(vval "Width")
HEIGHT=$(vval "Height")
FRAMERATE=$(vval "Frame rate")
END_FRAME=$(vval "Frame count")

[[ "$PROFILE" = "RA" ]] && FLAGS="${RA_CF}" || FLAGS="${AI_CF}"
[[ "$ALF" = "0" ]]  && FLAGS="${FLAGS} ${NOALF}" || FLAGS="${FLAGS} ${YESALF}"
[[ "$DB" = "0" ]] && FLAGS="${FLAGS} ${NODB}" || FLAGS="${FLAGS} ${YESDB}"
[[ "$SAO" = "0" ]] && FLAGS="${FLAGS} ${NOSAO}" || FLAGS="${FLAGS} ${YESSAO}"

SUFFIX="${PROFILE}_QP${QP}_ALF${ALF}_DB${DB}_SAO${SAO}"
DESTINATION="$(echo $FILE | cut -d "." -f 1 | cut -d "/" -f 2)_${SUFFIX}.vvc"
RECON_FILE="$(echo $FILE | cut -d "." -f 1 | cut -d "/" -f 2)_${SUFFIX}_rec.yuv"
LOG_FILE="$(echo $FILE | cut -d "." -f 1 | cut -d "/" -f 2)_${SUFFIX}.log"
echo "Processing ${DESTINATION}..."

${ENCODER_APP} ${ARGS} ${FLAGS} \
	--InputFile=$FILE \
	--BitstreamFile=$ODIR/$DESTINATION \
	--ReconFile=$ODIR/$RECON_FILE \
	-fr $FRAMERATE \
	-f $END_FRAME \
	-wdt $WIDTH \
	-hgt $HEIGHT \
	--QP=$QP > $ODIR/$LOG_FILE

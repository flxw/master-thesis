#!/bin/bash

#GPUNO=3
#LOGFILE=bpic2011
#FAMILY=sp2
OUTDIR="/home/felix.wolff2/docker_share/$LOGFILE"

if [ ! -d $OUTDIR ]; then
  mkdir "$OUTDIR"
fi

python model_runner.py $FAMILY individual --gpu=$GPUNO --output="$OUTDIR" ../logs/$LOGFILE/
python model_runner.py $FAMILY grouped --gpu=$GPUNO --output="$OUTDIR" ../logs/$LOGFILE/
python model_runner.py $FAMILY padded --gpu=$GPUNO --output="$OUTDIR" ../logs/$LOGFILE/
python model_runner.py $FAMILY windowed --gpu=$GPUNO --output="$OUTDIR" ../logs/$LOGFILE/

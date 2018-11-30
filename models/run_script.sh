#!/bin/bash

#GPUNO=3
#LOGFILE=bpic2011
#FAMILY=sp2

python model_runner.py $FAMILY individual --gpu=$GPUNO --output=/home/felix.wolff2/docker_share/$LOGFILE ../logs/$LOGFILE/
python model_runner.py $FAMILY grouped --gpu=$GPUNO --output=/home/felix.wolff2/docker_share/$LOGFILE ../logs/$LOGFILE/
python model_runner.py $FAMILY padded --gpu=$GPUNO --output=/home/felix.wolff2/docker_share/$LOGFILE ../logs/$LOGFILE/
python model_runner.py $FAMILY windowed --gpu=$GPUNO --output=/home/felix.wolff2/docker_share/$LOGFILE ../logs/$LOGFILE/

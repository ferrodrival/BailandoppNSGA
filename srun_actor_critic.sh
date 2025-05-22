#!/bin/sh
currenttime=`date "+%Y%m%d%H%M%S"`

if [ ! -d log ]; then
    mkdir log
fi

echo "[Usage] ./run.sh config_path [train|eval|visgt|anl|sample]"

# check config exists
if [ ! -e $1 ]; then
    echo "[ERROR] configuration file: $1 does not exist!"
    exit
fi

# Set experiment name based on config (optional: adjust as needed)
expname="actor_critic"

if [ ! -d ${expname} ]; then
    mkdir ${expname}
fi

echo "[INFO] saving results to, or loading files from: "$expname

PYTHONCMD="python -u main_actor_critic.py --config $1"

if [ "$2" = "train" ]; then
    $PYTHONCMD --train
elif [ "$2" = "eval" ]; then
    $PYTHONCMD --eval
elif [ "$2" = "visgt" ]; then
    $PYTHONCMD --visgt
elif [ "$2" = "anl" ]; then
    $PYTHONCMD --anl
elif [ "$2" = "sample" ]; then
    $PYTHONCMD --sample
else
    echo "[ERROR] Unknown mode: $2"
    echo "[Usage] ./run.sh config_path [train|eval|visgt|anl|sample]"
    exit 1
fi
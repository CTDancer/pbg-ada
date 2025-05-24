#!/bin/sh
export CUDA_VISIBLE_DEVICES=4,5,6,7

HOME_LOC=/home/tc415

STORAGE_LOC=/container_mount_storage/yz927

SCRIPT_LOC=$HOME_LOC/muPPIt/esmfold

LOG_LOC=$HOME_LOC/muPPIt/esmfold/logs

image=/home/yz927/containers/pure_centos_v4.sif

DATE=$(date +%m_%d)

singularity exec --bind /home/yz927/.local:/home/yz927/.local/ --nv $image bash -c " \

source /opt/rh/rh-python38/enable; \

export PYTHONPATH=/home/yz927/.local/lib/python3.8/site-packages:\$PYTHONPATH; \

python3.8 $SCRIPT_LOC/esmfold_ppiref.py \

" > esmfold_ppiref_test.log 2>&1

exit 0

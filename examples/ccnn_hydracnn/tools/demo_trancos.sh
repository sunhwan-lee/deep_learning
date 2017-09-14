#! /bin/bash

# Parameters
CONFIG_FILE=models/trancos/ccnn/ccnn_trancos_cfg.yml
TF_DATA=models/trancos/ccnn/trancos_ccnn.npy
TF_CLASS=models/trancos/ccnn/trancos_ccnn.py
TF_MODULE=../../caffe-tensorflow

#LOG="experiments/logs/trancos_ccnn_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
#exec &> >(tee -a "$LOG")
#echo Logging output to "$LOG"

# Time the task
T="$(date +%s)"

# Test Net
python code/test.py --tfdata ${TF_DATA} --tfclass ${TF_CLASS} --tfmodule ${TF_MODULE} --cfg ${CONFIG_FILE}

T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"

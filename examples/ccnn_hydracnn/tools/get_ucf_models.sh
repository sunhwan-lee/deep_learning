#! /bin/bash


# Download all the pretrained models using the UCF dataset

# Create dir
mkdir models/ucf
mkdir models/ucf/ccnn

# Download, untar, move and clean
for FOLD_NUM in 0 1 2 3 4
do
  # Form tar files names
  CCNN_TAR=ucf_ccnn_${FOLD_NUM}.caffemodel.tar.gz
  CCNN_MODEL=ucf_ccnn_${FOLD_NUM}.caffemodel
  
  # Get CCNN models
  wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/${CCNN_TAR}
  tar -zxvf ${CCNN_TAR}
  mv ${CCNN_MODEL} models/ucf/ccnn
  rm ${CCNN_TAR}

done

#! /bin/bash

# Download the pretrained models using the TRANCOS dataset.
wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/trancos_ccnn.caffemodel.tar.gz

# Untar
tar -zxvf trancos_ccnn.caffemodel.tar.gz

# Create dir and move file to the pretrained models dir
mkdir models/trancos
mkdir models/trancos/ccnn
mv trancos_ccnn.caffemodel models/trancos/ccnn

# Clean
rm trancos_ccnn.caffemodel.tar.gz

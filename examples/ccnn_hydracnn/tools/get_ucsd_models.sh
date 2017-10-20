#! /bin/bash


# Download all the pretrained models using the UCSD dataset

# Create dir
mkdir models/ucsd

# Download, untar, create dir, move and clean
mkdir models/ucsd/ccnn

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_ccnn_down.caffemodel.tar.gz
tar -zxvf ucsd_ccnn_down.caffemodel.tar.gz
mv ucsd_ccnn_down.caffemodel models/ucsd/ccnn
rm ucsd_ccnn_down.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_ccnn_max.caffemodel.tar.gz
tar -zxvf ucsd_ccnn_max.caffemodel.tar.gz
mv ucsd_ccnn_max.caffemodel models/ucsd/ccnn
rm ucsd_ccnn_max.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_ccnn_min.caffemodel.tar.gz
tar -zxvf ucsd_ccnn_min.caffemodel.tar.gz
mv ucsd_ccnn_min.caffemodel models/ucsd/ccnn
rm ucsd_ccnn_min.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_ccnn_up.caffemodel.tar.gz
tar -zxvf ucsd_ccnn_up.caffemodel.tar.gz
mv ucsd_ccnn_up.caffemodel models/ucsd/ccnn
rm ucsd_ccnn_up.caffemodel.tar.gz
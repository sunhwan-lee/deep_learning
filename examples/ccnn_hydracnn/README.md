# Towards perspective-free object counting with deep learning

By [Daniel Oñoro-Rubio](https://es.linkedin.com/in/daniel-oñoro-71062756) and [Roberto J. López-Sastre](http://agamenon.tsc.uah.es/Personales/rlopez/).

GRAM, University of Alcalá, Alcalá de Henares, Spain.

This is the `TensorFlow` implementation of the work described in [ECCV 2016 paper](http://agamenon.tsc.uah.es/Investigacion/gram/publications/eccv2016-onoro.pdf). 

This repository provides the implementation of CCNN and Hydra models for object counting.

## Contents
1. [Requirements: software](#requirements-software)
2. [Basic installation](#basic-installation-sufficient-for-the-demo)
3. [Demo](#demo)
4. [How to reproduce the results of the paper](#how-to-reproduce-the-results-of-the-paper)
5. [Remarks](#remarks)
6. [Acknowledgements](#acknowledgements)

### Requirements: software

1. Developed and tested the code on mac osX Sierra 10.12.6.

2. Requirements for `Tensorflow`. Follow the [Tensorflow installation instructions](https://www.tensorflow.org/install/).

3. For windows, [Cygwin](https://cygwin.com/install.html) is required. Make sure that `wget` package is selected.

### Convert Caffe model to TensorFlow

[caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) is used to convert caffe model to TensorFlow model because the original code was written in caffe framework.

* Example
```Shell
python caffe-tensorflow/convert.py --caffemodel examples/ccnn_hydracnn/models/trancos/ccnn/trancos_ccnn.caffemodel --data-output-path examples/ccnn_hydracnn/models/trancos/ccnn/trancos_ccnn.npy --code-output-path examples/ccnn_hydracnn/models/trancos/ccnn/trancos_ccnn.py examples/ccnn_hydracnn/models/trancos/ccnn/ccnn_deploy.prototxt
```

### Demo

We here provide a demo consisting in predicting the number of vehicles in the test images of the [TRANCOS dataset](http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/), which was used in our ECCV paper. 

This demo uses the CCNN model described in the paper. The results reported in the paper can be reproduced with this demo.

To run the demo, these are the steps to follow:

1. Download the TRANCOS dataset by executing the following script provided:
	```Shell
	./tools/get_trancos.sh
	```

2. You must have now a new directory with the TRANCOS dataset in the path `data/TRANCOS`.

3. Download the TRANCOS CCNN pretrained model.
	```Shell
	./tools/get_trancos_model.sh
	```

4. Finally, to run the demo, simply execute the following command:
	```Shell
	./tools/demo.sh
	```

### How to reproduce the results of the paper?

We provide here the scripts needed to **train** and **test** all the models (CCNN and Hydra) with the datasets used in our ECCV paper. These are the steps to follow.

#### Download a dataset

In order to download and setup a dataset we recommend to use our scripts. To do so, just place yourself in the $PROJECT directory and run one of the following scripts:

* [TRANCOS dataset](http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/):
 
	```Shell
    ./tools/get_trancos.sh
    ```

* [UCSD dataset](http://www.svcl.ucsd.edu/projects/peoplecnt/):

	```Shell
    ./tools/get_ucsd.sh
    ```

* [UCF dataset](http://crcv.ucf.edu/data/crowd_counting.php):

	```Shell
    ./tools/get_ucf.sh
    ```

**Note:** Make sure the folder "data/" does not already contain the dataset.


#### Download pre-trained models

All our pre-trained models can be downloaded using the corresponding script:

	```Shell
    ./tools/get_all_DATASET_CHOSEN_models.sh
    ```
Simply substitute DATASET_CHOSEN by: trancos, ucsd or ucf.

#### Test the pretrained models
1. Edit the corresponding script $PROJECT/experiments/scripts/DATASET_CHOSEN_test_pretrained.sh

2. Run the corresponding scripts.

       ```Shell
    ./experiments/scripts/DATASET_CHOSEN_test_pretrained.sh
    ```
Note that this pretrained models will let you reproduce the results in our paper.


#### Train/test the model chosen

1. Edit the launching script (e.g.: $PROJECT/experiments/scripts/DATASET_CHOSEN_train_test.sh).

2. Place you in $PROJECT folder and run the launching script by typing:

	```Shell
    ./experiments/scripts/DATASET_CHOSEN_train_test.sh
    ```


### Remarks

In order to provide a better distribution, this repository *unifies and reimplements* in Python some of the original modules. Due to these changes in the libraries used, the results produced by this software might be slightly different from the ones reported in the paper.


### Acknowledgements
This work is supported by the projects of the DGT with references SPIP2014-1468 and SPIP2015-01809, and the project of the MINECO TEC2013-45183-R.

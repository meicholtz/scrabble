# YAD2K: Yet Another Darknet 2 Keras

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

YAD2K is a 90% Keras/10% Tensorflow implementation of YOLO_v2.

Original paper: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) by Joseph Redmond and Ali Farhadi.

![YOLO_v2 COCO model with test_yolo defaults](etc/dog_small.jpg)

--------------------------------------------------------------------------------

## Requirements

- [Keras](https://github.com/fchollet/keras)
- [Tensorflow](https://www.tensorflow.org/)
- [NumPy](http://www.numpy.org/)
- [h5py](http://www.h5py.org/) (For Keras model serialization.)
- [Pillow](https://pillow.readthedocs.io/) (For rendering test results.)
- [Python 3](https://www.python.org/)
- [pydot-ng](https://github.com/pydot/pydot-ng) (Optional for plotting model.)

## Installation
```bash
git clone https://github.com/allanzelener/yad2k.git
cd yad2k

# [Option 1] To replicate the conda environment:
conda env create -f environment.yml
source activate yad2k
# [Option 2] Install everything globaly.
pip install numpy h5py pillow
pip install tensorflow-gpu  # CPU-only: conda install -c conda-forge tensorflow
pip install keras # Possibly older release: conda install keras
```

## Quick Start

- Download Darknet model cfg and weights from the [official YOLO website](http://pjreddie.com/darknet/yolo/).
- Convert the Darknet YOLO_v2 model to a Keras model.
- Test the converted model on the small test set in `images/`.

```bash
wget http://pjreddie.com/media/files/yolo.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
./yad2k.py yolo.cfg yolo.weights model_data/yolo.h5
./test_yolo.py model_data/yolo.h5  # output in images/out/
```

See `./yad2k.py --help` and `./test_yolo.py --help` for more options.

--------------------------------------------------------------------------------

## More Details

The YAD2K converter currently only supports YOLO_v2 style models, this include the following configurations: `darknet19_448`, `tiny-yolo-voc`, `yolo-voc`, and `yolo`.

`yad2k.py -p` will produce a plot of the generated Keras model. For example see [yolo.png](etc/yolo.png).

YAD2K assumes the Keras backend is Tensorflow. In particular for YOLO_v2 models with a passthrough layer, YAD2K uses `tf.space_to_depth` to implement the passthrough layer. The evaluation script also directly uses Tensorflow tensors and uses `tf.non_max_suppression` for the final output.

`voc_conversion_scripts` contains two scripts for converting the Pascal VOC image dataset with XML annotations to either HDF5 or TFRecords format for easier training with Keras or Tensorflow.

`yad2k/models` contains reference implementations of Darknet-19 and YOLO_v2.

`train_overfit` is a sample training script that overfits a YOLO_v2 model to a single image from the Pascal VOC dataset.

## Known Issues and TODOs

- Expand sample training script to train YOLO_v2 reference model on full dataset.
- Support for additional Darknet layer types.
- Tuck away the Tensorflow dependencies with Keras wrappers where possible.
- YOLO_v2 model does not support fully convolutional mode. Current implementation assumes 1:1 aspect ratio images.

## Darknets of Yore

YAD2K stands on the shoulders of giants.

- :fire: [Darknet](https://github.com/pjreddie/darknet) :fire:
- [Darknet.Keras](https://github.com/sunshineatnoon/Darknet.keras) - The original D2K for YOLO_v1.
- [Darkflow](https://github.com/thtrieu/darkflow) - Darknet directly to Tensorflow.
- [caffe-yolo](https://github.com/xingwangsfu/caffe-yolo) - YOLO_v1 to Caffe.
- [yolo2-pytorch](https://github.com/longcw/yolo2-pytorch) - YOLO_v2 in PyTorch.

## **(NEW 7/31/2018)** How to run YOLO for 2D images?

**STEP 1: Generate synthetic 2D images (only if needed)**

Run the following script in MATLAB:

```
makeSyntheticImages2
```

Make sure to edit the variables in "Set parameters" section to suit your needs. The output is a directory of png images and text files. Each text file contains bounding boxes for the corresponding image in the format expected by yad2k:

	class1 xmin1 ymin1 xmax1 ymax1
	class2 xmin2 ymin2 xmax2 ymax2
	...
	classN xminN yminN xmaxN ymaxN

Currently, this script supports circles, squares, and triangles, but it can be expanded with minimal effort to include other shapes.

**STEP 2: Convert images to numpy array**

Run the following python script from the command line:

```
python utils/convert_images_to_npz.py INPUT_DIRECTORY
```

Replace INPUT_DIRECTORY with the path where the data is saved (this would be `saveroot` in `makeSyntheticImages2.m`). The output will be a npz file in the parent directory of INPUT_DIRECTORY. For example, if the input is stored in the folder

yad2k/images/shapes

then the output will be stored in the file

yad2k/images/shapes.npz

**STEP 3: Train YOLO network**

Run the following python script from the command line:

```
python train_yolo.py
```

Make sure to set the optional arguments accordingly and specify a GPU, if desired. For example,

```
CUDA_VISIBLES_DEVICES=0 python train_yolo.py -a ANCHORS_PATH -c CLASSES_PATH -d DATA_PATH -o OUTPUT_PATH
```

**STEP 4: Test YOLO network**

Run the following python script from the command line:

```
python test_yolo.py MODEL_PATH DATA_PATH
```

where MODEL_PATH is the h5 file that was saved during training, and DATA_PATH is the npz file containing test data (you could run `makeSyntheticImages2.m` again to generate new data for testing). Make sure to set the optional arguments accordingly and specify a GPU, if desired. For example,

```
CUDA_VISIBLES_DEVICES=0 python test_yolo.py MODEL_PATH DATA_PATH -a ANCHORS_PATH -c CLASSES_PATH -o OUTPUT_PATH -b BATCH
```

Currently, the output of the network is saved as a mat file for further processing in MATLAB.

**STEP 5: Evaluate test results**

Run the following scripts in MATLAB:

```
computeAveragePrecision2
viewSampleDetections2
```

## **(NEW 7/31/2018)** How to run YOLO for 3D images?

NOTE: The pipeline for 3D is *very* similar to 2D. In most cases, functions/scripts with a '2' or '2d' at the end are replaced by a '3' or '3d', respectively.

**STEP 1: Generate synthetic 3D images (only if needed)**

Run the following script in MATLAB:

```
makeSyntheticImages3
```

Make sure to edit the variables in "Set parameters" section to suit your needs. The output is a mat file containing the images and bounding boxes.

Currently, this script only supports spheres, but it can be expanded with moderate effort to include other shapes.

**STEP 2: Convert mat file to numpy array**

Run the following python script from the command line:

```
python utils/convert_mat_to_npz.py INPUT OUTPUT
```

where INPUT points to the mat file created from **STEP 1** and OUTPUT points to the npz file where the data will be saved.

**STEP 3: Train YOLO network**

Run the following python script from the command line:

```
python train_yolo_3d.py
```

Make sure to set the optional arguments accordingly and specify a GPU, if desired. For example,

```
CUDA_VISIBLES_DEVICES=0 python train_yolo_3d.py -a ANCHORS_PATH -c CLASSES_PATH -d DATA_PATH -o OUTPUT_PATH
```

**STEP 4: Test YOLO network**

Run the following python script from the command line:

```
python test_yolo_3d.py MODEL_PATH DATA_PATH
```

where MODEL_PATH is the h5 file that was saved during training, and DATA_PATH is the npz file containing test data (you could run `makeSyntheticImages3.m` again to generate new data for testing). Make sure to set the optional arguments accordingly and specify a GPU, if desired. For example,

```
CUDA_VISIBLES_DEVICES=0 python test_yolo_3d.py MODEL_PATH DATA_PATH -a ANCHORS_PATH -c CLASSES_PATH -o OUTPUT_PATH -b BATCH
```

Currently, the output of the network is saved as a mat file for further processing in MATLAB.

**STEP 5: Evaluate test results**

Run the following scripts in MATLAB:

```
computeAveragePrecision3
viewSampleDetections3
```

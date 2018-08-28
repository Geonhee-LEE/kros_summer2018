## Object Detection by YOLO

Tensorflow implementation of [YOLO](https://arxiv.org/pdf/1506.02640.pdf), including training and test phase.

This implementation of YOLO are simple refactoring of [Peng Zhang's works](https://github.com/hizhangp/yolo_tensorflow) for lab.

### Test Phase

1. Download [YOLO_small](https://drive.google.com/file/d/1ZRFgjtfFPVxn_nqWrs4Fq3ARNkjTDbHa/view?usp=sharing)
weight file and put it in `data/weights`

2. Run `test.py`
	```Shell
	$ python3 test.py
	```

### Training Phase

1. Download [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and extract it to `data/pascal_voc` director for training.

2. Run `train.py`
	```Shell
	$ python3 train.py
	```

### Requirements
- tensorflow
- numpy
- matplotlib
- opencv-python

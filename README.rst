Self-supervision
================

Training and evaluating self-supervised deep neural networks.

* http://people.cs.uchicago.edu/~larsson/color-proxy

Models
------

These models were all trained entirely without labeled data using colorization as a proxy task:

* `AlexNet <http://people.cs.uchicago.edu/~larsson/color-proxy/models/alexnet.caffemodel.h5>`_
* `VGG-16 <http://people.cs.uchicago.edu/~larsson/color-proxy/models/vgg16.caffemodel.h5>`_
* `ResNet-152 <http://people.cs.uchicago.edu/~larsson/color-proxy/models/resnet152.caffemodel.h5>`_

The models are saved in Caffe HDF5 files. Our evaluation code in TensorFlow can load these files. The AlexNet model uses LRN (applied before pooling)

Paper
-----

* *Colorization as a Proxy Task for Visual Understanding*, Larsson, Maire, Shakhnarovich, CVPR 2017 (https://arxiv.org/abs/1703.04044)

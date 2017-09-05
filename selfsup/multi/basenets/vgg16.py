from __future__ import division, print_function, absolute_import
from .basenet import BaseNet
import selfsup

MEAN = 114.154 / 255.0

class VGG16Net(BaseNet):
    def __init__(self, use_batch_norm=False):
        self._use_batch_norm = use_batch_norm

    @property
    def name(self):
        return 'vgg'

    @property
    def canonical_input_size(self):
        return 224

    @property
    def hypercolumn_layers(self):
        return [
            ('conv1_1', 1.0),
            ('conv1_2', 1.0),
            ('conv2_1', 2.0),
            ('conv2_2', 2.0),
            ('conv3_1', 4.0),
            ('conv3_2', 4.0),
            ('conv3_3', 4.0),
            ('conv4_1', 8.0),
            ('conv4_2', 8.0),
            ('conv4_3', 8.0),
            ('conv5_1', 16.0),
            ('conv5_2', 16.0),
            ('conv5_3', 16.0),
            ('fc6', 32.0),
            ('fc7', 32.0),
        ]

    def build(self, x01, phase_test, global_step, settings={}):
        info = selfsup.info.create(scale_summary=True)
        x = x01 - MEAN
        if self._use_batch_norm:
            z = selfsup.model.vgg16_bn.build_network(x, info=info,
                    convolutional=settings.get('convolutional', False),
                    final_layer=False,
                    squeezed=True,
                    global_step=global_step,
                    phase_test=phase_test)
        else:
            z = selfsup.model.vgg16.build_network(x, info=info,
                    convolutional=settings.get('convolutional', False),
                    final_layer=False,
                    squeezed=True,
                    phase_test=phase_test)
        info['activations']['x01'] = x01
        info['activations']['x'] = x
        info['activations']['top'] = z
        info['weights']['firstconv:weights'] = info['weights']['conv1_1:weights']
        info['weights']['firstconv:biases'] = info['weights']['conv1_1:biases']
        return info

    def save_caffemodel(self, path, session, verbose=False, prefix=''):
        layers = [
            'conv1_1',
            'conv1_2',
            'conv2_1',
            'conv2_2',
            'conv3_1',
            'conv3_2',
            'conv3_3',
            'conv4_1',
            'conv4_2',
            'conv4_3',
            'conv5_1',
            'conv5_2',
            'conv5_3',
            ('fc6', 'fc6_pre'),
            ('fc7', 'fc7_pre'),
        ]

        tr = {'fc6': (4096, 512, 7, 7)}

        selfsup.caffe.save_caffemodel(path, session, layers, prefix=prefix,
                                 conv_fc_transitionals=tr,
                                 save_batch_norm=True,
                                 color_layer='conv1_1', verbose=verbose)

from __future__ import division, print_function, absolute_import
from .basenet import BaseNet
import selfsup.model.resnet152_bn
import selfsup

MEAN = 114.154 / 255.0

class ResNet152(BaseNet):
    def __init__(self, use_batch_norm=True):
        assert use_batch_norm is True, "Must use with batch norm"
        pass

    @property
    def name(self):
        return 'resnet'

    @property
    def canonical_input_size(self):
        return 224

    @property
    def hypercolumn_layers(self):
        return [
            ('conv1', 2.0),
            ('res2a', 4.0),
            ('res2b', 4.0),
            ('res2c', 4.0),
            ('res3b1', 8.0),
            ('res3b4', 8.0),
            ('res3b7', 8.0),
            ('res4b5', 16.0),
            ('res4b10', 16.0),
            ('res4b15', 16.0),
            ('res4b20', 16.0),
            ('res4b25', 16.0),
            ('res4b30', 16.0),
            ('res4b35', 16.0),
            ('res5c', 32.0),
        ]

    def build(self, x01, phase_test, global_step, settings={}):
        info = selfsup.info.create(scale_summary=True)
        #info['config']['return_weights'] = True
        x = x01 - MEAN
        z = selfsup.model.resnet152_bn.build_network(x, info=info,
                convolutional=settings.get('convolutional', False),
                final_layer=False,
                #squeezed=True,
                phase_test=phase_test)
        info['activations']['x01'] = x01
        info['activations']['x'] = x
        info['activations']['top'] = z
        info['weights']['firstconv:weights'] = info['weights']['conv1:weights']
        info['weights']['firstconv:biases'] = None #info['weights']['conv1:biases']
        return info

    def save_caffemodel(self, path, session, verbose=False, prefix=''):

        layers = [
            ('conv1', 2.0),
            ('res2a', 4.0),
            ('res2b', 4.0),
            ('res2c', 4.0),
            ('res3b1', 8.0),
            ('res3b4', 8.0),
            ('res3b7', 8.0),
            ('res4b5', 16.0),
            ('res4b10', 16.0),
            ('res4b15', 16.0),
            ('res4b20', 16.0),
            ('res4b25', 16.0),
            ('res4b30', 16.0),
            ('res4b35', 16.0),
            ('res5c', 32.0),
        ]

        tr = {}#'fc6': (4096, 256, 7, 7)}

        selfsup.caffe.save_caffemodel(path, session, layers, prefix=prefix,
                                 conv_fc_transitionals=tr,
                                 color_layer='conv1_1', verbose=verbose)

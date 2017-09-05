from __future__ import division, print_function, absolute_import
from .basenet import BaseNet
import selfsup
from selfsup.util import DummyDict
import selfsup.alex_bn2
import selfsup.model.alex
import selfsup.model.alex_bn
import selfsup.model.alex_bn2
import selfsup.model.alex_transposed

MEAN = 114.154 / 255.0

class AlexNet(BaseNet):
    def __init__(self, use_batch_norm=False, use_lrn=False):
        self._use_batch_norm = use_batch_norm
        self._use_lrn = use_lrn

    @property
    def name(self):
        return 'alex'

    @property
    def canonical_input_size(self):
        return 227

    @property
    def hypercolumn_layers(self):
        return [
            ('x',     1.0),
            ('conv1', 4.0),
            ('conv2', 8.0),
            ('conv3', 16.0),
            ('conv4', 16.0),
            ('conv5', 16.0),
            ('fc6', 32.0),
            ('fc7', 32.0),
        ]

    def build(self, x01, phase_test, global_step, settings={}):
        self._global_step = global_step
        self._phase_test = phase_test

        info = selfsup.info.create(scale_summary=True)
        info['config']['save_pre'] = True
        x = x01 - MEAN
        if self._use_batch_norm == 'v2':
            z = selfsup.model.alex_bn2.build_network(x, info=info,
                    convolutional=settings.get('convolutional', False),
                    final_layer=False,
                    well_behaved_size=False,
                    use_lrn=self._use_lrn,
                    global_step=global_step,
                    phase_test=phase_test)
        elif self._use_batch_norm:
            z = selfsup.model.alex_bn.build_network(x, info=info,
                    convolutional=settings.get('convolutional', False),
                    final_layer=False,
                    well_behaved_size=False,
                    use_lrn=self._use_lrn,
                    global_step=global_step,
                    phase_test=phase_test)
        else:
            z = selfsup.model.alex.build_network(x, info=info,
                    convolutional=settings.get('convolutional', False),
                    final_layer=False,
                    well_behaved_size=False,
                    use_lrn=self._use_lrn,
                    phase_test=phase_test)

        info['activations']['x01'] = x01
        info['activations']['x'] = x
        info['activations']['top'] = z
        info['weights']['firstconv:weights'] = info['weights']['conv1:weights']
        info['weights']['firstconv:biases'] = info['weights']['conv1:biases']
        return info

    def decoder(self, z, channels=1, multiple=4, from_name=None, settings=DummyDict(), info=DummyDict()):
        settings = dict(phase_test=self._phase_test,
                        global_step=self._global_step,
                        use_batch_norm=self._use_batch_norm,
                        info=info)

        if from_name is not None:
            settings['from_name'] = from_name

        if multiple == 4:
            y = selfsup.model.alex_transposed.decoder(z, to_name='conv2', **settings)

            sh = [z.get_shape().as_list()[0], 57, 57, channels]
            y = selfsup.ops.upconv(y, sh[-1], size=5, strides=2, name='upconv1', output_shape=sh, padding='VALID')
            y = y[:, 0:-1, 0:-1]
        elif multiple == 1:
            y = selfsup.model.alex_transposed.decoder(z, **settings)
        else:
            raise ValueError("Multiple not supported")

        return y

    def save_caffemodel(self, path, session, verbose=False, prefix=''):
        layers = [
            'conv1',
            'conv2',
            'conv3',
            'conv4',
            'conv5',
            'fc6',
            'fc7',
        ]

        tr = {'fc6': (4096, 256, 6, 6)}

        selfsup.caffe.save_caffemodel(path, session, layers, prefix=prefix,
                                 conv_fc_transitionals=tr,
                                 save_batch_norm=True,
                                 color_layer='conv1', verbose=verbose)

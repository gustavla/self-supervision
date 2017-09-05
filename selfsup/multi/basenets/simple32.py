from .basenet import BaseNet
from selfsup.util import DummyDict
import selfsup.model.cnet
import selfsup

class Simple32(BaseNet):
    def __init__(self, use_batch_norm=False, thin_top=False):
        self._use_batch_norm = use_batch_norm
        self._thin_top = thin_top

    @property
    def name(self):
        return 'simple32'

    @property
    def canonical_input_size(self):
        return 32

    @property
    def hypercolumn_layers(self):
        return [
            #('x',     1.0),
            ('conv1', 1.0),
            ('conv1b', 1.0),
            ('conv2', 2.0),
            ('conv2b', 2.0),
            ('conv3', 4.0),
            ('conv3b', 4.0),
            ('conv4', 8.0),
            ('conv4b', 8.0),
            #('fc6',   32.0),
        ]

    def build(self, x, phase_test, global_step, settings={}):
        self._global_step = global_step
        self._phase_test = phase_test

        info = selfsup.info.create(scale_summary=True)
        info['config']['save_pre'] = True
        z = selfsup.model.cnet.build_network_small(x, info=info,# parameters=data,
                                 convolutional=False, final_layer_channels=None,
                                 phase_test=phase_test,
                                 use_dropout=True,
                                 use_batch_norm=self._use_batch_norm,
                                 global_step=global_step,
                                 top_channels=100 if self._thin_top else None,
                                 )

        info['activations']['x'] = x
        info['activations']['top'] = z
        info['weights']['firstconv:weights'] = info['weights']['conv1:weights']
        info['weights']['firstconv:biases'] = info['weights']['conv1:biases']
        return info

    def decoder(self, z, channels=1, multiple=4, from_name=None, settings=DummyDict(), info=DummyDict()):
        raise NotImplemented()

    def save_caffemodel(self, path, session, verbose=False, prefix=''):
        selfsup.model.cnet.save_caffemodel(path, session, prefix=prefix)

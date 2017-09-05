

class BaseNet:
    def __init__(self):
        pass

    @property
    def name(self):
        raise NotImplemented('Abstract base class')

    @property
    def canonical_input_size(self):
        raise NotImplemented('Abstract base class')

    @property
    def hypercolumn_layers(self):
        """Suggested hypercolumn layers"""
        return []

    @property
    def layers(self):
        return [name for name, _ in self.hypercolumn_layers]

    def build(self):
        raise NotImplemented('Abstract base class')

    def save_caffemodel(self, path, session, verbose=False):
        raise NotImplemented('Abstract base class')

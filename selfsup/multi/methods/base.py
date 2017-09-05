


class Method:
    def __init__(self, name, batch_size):
        self.name = name
        self.batch_size = batch_size

    def input_size(self):
        return 0

    def batch_loader(self):
        raise NotImplemented("Abstract base class")

    def build_network(network, y, phase_test):
        raise NotImplemented("Abstract base class")

    @property
    def basenet_settings():
        return {}

    def make_path(self, name, ext, iteration):
        return f'img/{self.name}_{name}_{iteration}.png'

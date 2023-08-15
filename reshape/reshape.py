from functional.chunk_reshape import external_reshape
from functional.chunk_reshape import external_reshape2
from functional.chunk_reshape import external_reshape3
from torch import nn


class MyReshape:
    def __init__(self, device):
        self.device = device

    def __call__(self, x, map_type=1):
        return external_reshape(x, map_type, self.device)


class MyReshape2:
    def __init__(self, device):
        self.device = device

    def __call__(self, x, map_type=1):
        return external_reshape2(x, map_type, self.device)


class MyReshape3:
    def __init__(self, device):
        self.device = device

    def __call__(self, x, map_type=1):
        return external_reshape3(x, map_type, self.device)

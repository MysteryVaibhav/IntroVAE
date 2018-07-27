import torch
import numpy as np


class Util:
    def __init__(self, params):
        super(Util, self).__init__()
        self.params = params

    @staticmethod
    def to_tensor(array):
        return torch.from_numpy(np.array(array)).float()

    def to_variable(self, tensor, requires_grad=False):
        if self.params.cuda:
            tensor = tensor.cuda()
        return torch.autograd.Variable(tensor, requires_grad=requires_grad)

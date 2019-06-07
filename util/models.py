import sys
import torch
import torch.nn as nn

import numpy as np

from abc import ABCMeta, abstractmethod
from .modules import FCLayer, ReLULayer, SigdLayer, TanhLayer, ArctanLayer
from .utility import reduced_bv_bm, reduced_bm_bv

str2func = {'relu': ReLULayer, 'tanh': TanhLayer, 'sigd': SigdLayer, 'arctan': ArctanLayer}

class Model(nn.Module, metaclass = ABCMeta):

    def __init__(self,):

        super(Model, self).__init__()

    @abstractmethod
    def forward(self, x):
        '''
        >>> do forward pass with a given input
        '''

        raise NotImplementedError

    @abstractmethod
    def bound(self, x, ori_perturb_norm = np.inf, ori_perturb_eps = None):
        '''
        >>> do bound calculation

        >>> x: the input data point
        >>> ori_perturb_norm: float, the norm defined as adversarial budget
        >>> ori_perturb_eps: tensor of shape [batch_size, in_dim]
        '''

        raise NotImplementedError

class MLP(Model):

    def __init__(self, in_dim, hidden_dims, out_dim, nonlinearity = 'relu'):
        '''
        >>> in_dim: int, the input dimension
        >>> hidden_dims: list, the list of hidden dimensions
        >>> out_dim: int, the output dimension
        >>> nonlinearity: str, activation function
        '''

        super(Model, self).__init__()

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.nonlinearity = nonlinearity
        self.neurons = [in_dim,] + hidden_dims + [out_dim,]

        self.main_block = nn.Sequential()

        for idx, (in_neuron, out_neuron) in enumerate(zip(self.neurons[:-2], self.neurons[1:-1])):
            linear_layer = FCLayer(in_features = in_neuron, out_features = out_neuron)
            nonlinear_layer = str2func[self.nonlinearity]()
            self.main_block.add_module('layer_%d'%idx, linear_layer)
            self.main_block.add_module('nonlinear_%d'%idx, nonlinear_layer)

        self.output = FCLayer(in_features = self.neurons[-2], out_features = self.neurons[-1])

    def forward(self, x):

        out = x.view(x.size(0), -1)
        out = self.main_block(out)
        out = self.output(out)

        return out

    def bound(self, x, ori_perturb_norm = np.inf, ori_perturb_eps = None):

        x = x.view(x.size(0), -1)

        W_list = [torch.ones_like(x),]
        m1_list = [x,]
        m2_list = [x,]
        l = x
        u = x

        # Main Block
        for idx, layer in enumerate(self.main_block):

            l, u, W_list, m1_list, m2_list = layer.bound(l = l, u = u, W_list = W_list, m1_list = m1_list, m2_list = m2_list,
                ori_perturb_norm = ori_perturb_norm, ori_perturb_eps = ori_perturb_eps, first_layer = idx == 0)

        # Output 
        l, u, W_list, m1_list, m2_list = self.output.bound(l = l, u = u, W_list = W_list, m1_list = m1_list, m2_list = m2_list,
            ori_perturb_norm = ori_perturb_norm, ori_perturb_eps = ori_perturb_eps, first_layer = len(self.hidden_dims) == 0)

        return l, u


'''
>>> This file creates modules that can do forward, backward pass as well as bound propagation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from abc import ABCMeta, abstractmethod

import numpy as np

from .linearize import linearize_relu, linearize_sigd, linearize_tanh, linearize_arctan
from .utility import reduced_m_bm, reduced_bm_bm, reduced_bv_bm, reduced_bm_bv, quad_bound_calc

class Layer(nn.Module, metaclass = ABCMeta):

    def __init__(self,):

        super(Layer, self).__init__()

    @abstractmethod
    def forward(self, x):
        '''
        >>> do forward pass with a given input
        '''

        raise NotImplementedError

    @abstractmethod
    def bound(self, l, u, W_list, m1_list, m2_list, ori_perturb_norm = None, ori_perturb_eps = None, first_layer = False):
        '''
        >>> do bound calculation

        >>> l, u: the lower and upper bound of the input, of shape [batch_size, immediate_in_dim]
        >>> W_list: the transformation matrix introduced by the previous layers, of shape [batch_size, out_dim, in_dim]
        >>> m1_list, m2_list: the bias introduced by the previous layers, of shape [batch_size, in_dim]
        >>> ori_perturb_norm, ori_perturb_eps: the original perturbation, default is None
        >>> first_layer: boolean, whether or not this layer is the first layer
        '''

        raise NotImplementedError


class FCLayer(Layer):

    def __init__(self, in_features, out_features):

        super(FCLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):

        return F.linear(x, self.layer.weight, self.layer.bias)

    def bound(self, l, u, W_list, m1_list, m2_list, ori_perturb_norm = None, ori_perturb_eps = None, first_layer = False):

        batch_size = l.shape[0]

        # quad method
        # if the bias term in the last iteration is the same, we can merge it with the current one
        max_var = torch.max(torch.abs(m1_list[-1] - m2_list[-1]))

        # Update the transition weight matrix
        update_list = W_list if max_var > 1e-4 or ori_perturb_norm != None else W_list[:-1]
        for idx, W in enumerate(update_list):
            W_list[idx] = reduced_m_bm(self.layer.weight, W)

        # Add the contribution of this layer
        if max_var > 1e-4 or ori_perturb_norm != None:
            W_list.append(torch.ones([batch_size, self.out_features], device = self.layer.weight.device))
            m1_list.append(self.layer.bias.unsqueeze(0).repeat(batch_size, 1))
            m2_list.append(self.layer.bias.unsqueeze(0).repeat(batch_size, 1))
        else:
            W_list[-1] = torch.ones([batch_size, self.out_features], device = self.layer.weight.device)
            m1_list[-1] = torch.matmul(m1_list[-1], self.layer.weight.transpose(0, 1)) + self.layer.bias
            m2_list[-1] = torch.matmul(m2_list[-1], self.layer.weight.transpose(0, 1)) + self.layer.bias

        quad_low_bound, quad_up_bound = quad_bound_calc(W_list, m1_list, m2_list, ori_perturb_norm, ori_perturb_eps)

        # simp method
        if first_layer == True:
            primal_norm = ori_perturb_norm
            dual_norm = 1. / (1. - 1. / primal_norm)
            adjust = torch.norm(self.layer.weight.unsqueeze(0) * ori_perturb_eps.unsqueeze(1), dim = 2, p = dual_norm)  # of shape [batch_size, out_dim]
        else:
            adjust = 0.

        W_neg = torch.clamp(self.layer.weight, max = 0.)
        W_pos = torch.clamp(self.layer.weight, min = 0.)

        simp_low_bound = l.matmul(W_pos.t()) + u.matmul(W_neg.t()) - adjust + self.layer.bias
        simp_up_bound = l.matmul(W_neg.t()) + u.matmul(W_pos.t()) + adjust + self.layer.bias

        low_bound = torch.max(quad_low_bound, simp_low_bound)
        up_bound = torch.min(quad_up_bound, simp_up_bound)

        return low_bound, up_bound, W_list, m1_list, m2_list

class ReLULayer(Layer):

    def __init__(self,):

        super(ReLULayer, self).__init__()

    def forward(self, x):

        return F.relu(x, inplace = True)

    def bound(self, l, u, W_list, m1_list, m2_list, ori_perturb_norm = None, ori_perturb_eps = None, first_layer = False):

        assert first_layer == False, 'the first layer cannot be ReLU'

        batch_size = l.shape[0]

        # quad method
        # Obtain D, m1, m2
        D, m1, m2 = linearize_relu(l, u)
        D = D.reshape(batch_size, -1)               # of shape [batch_size, dim]
        m1 = m1.reshape(batch_size, -1)             # of shape [batch_size, dim]
        m2 = m2.reshape(batch_size, -1)             # of shape [batch_size, dim]
        out_dim = D.shape[1]

        # For potential merge
        max_var = torch.max(torch.abs(m1_list[-1] - m2_list[-1]))

        # Update
        update_list = W_list if max_var > 1e-4 else W_list[:-1]
        for idx, W in enumerate(update_list):
            W_list[idx] = reduced_bm_bm(D, W)

        # Add the contribution of this layer
        if max_var > 1e-4:
            W_list.append(torch.ones([batch_size, out_dim], device = D.device))
            m1_list.append(m1)
            m2_list.append(m2)
        else:
            m1_list[-1] = m1_list[-1] * D + m1
            m2_list[-1] = m2_list[-1] * D + m2

        quad_low_bound, quad_up_bound = quad_bound_calc(W_list, m1_list, m2_list, ori_perturb_norm, ori_perturb_eps)

        # simp method
        simp_low_bound = F.relu(l, inplace = True)
        simp_up_bound = F.relu(u, inplace = True)

        low_bound = torch.max(quad_low_bound, simp_low_bound)
        up_bound = torch.min(quad_up_bound, simp_up_bound)

        return low_bound, up_bound, W_list, m1_list, m2_list

class SigdLayer(Layer):

    def __init__(self,):

        super(SigdLayer, self).__init__()

    def forward(self, x):

        return F.sigmoid(x)

    def bound(self, l, u, W_list, m1_list, m2_list, ori_perturb_norm = None, ori_perturb_eps = None, first_layer = False):

        assert first_layer == False, 'the first layer cannot be ReLU'

        batch_size = l.shape[0]

        # quad method
        # Obtain D, m1, m2
        D, m1, m2 = linearize_sigd(l, u)
        D = D.reshape(batch_size, -1)
        m1 = m1.reshape(batch_size, -1)
        m2 = m2.reshape(batch_size, -1)
        out_dim = D.shape[1]

        # For potential merge
        max_var = torch.max(torch.abs(m1_list[-1] - m2_list[-1]))

        # Update
        update_list = W_list if max_var > 1e-4 else W_list[:-1]
        for idx, W in enumerate(update_list):
            W_list[idx] = reduced_bm_bm(D, W)

        # Add the contribution of this layer
        if max_var > 1e-4:
            W_list.append(torch.ones([batch_size, out_dim], device = D.device))
            m1_list.append(m1)
            m2_list.append(m2)
        else:
            m1_list[-1] = m1_list[-1] * D + m1
            m2_list[-1] = m2_list[-1] * D + m2

        quad_low_bound, quad_up_bound = quad_bound_calc(W_list, m1_list, m2_list, ori_perturb_norm, ori_perturb_eps)

        # simp method
        simp_low_bound = F.sigmoid(l)
        simp_up_bound = F.sigmoid(u)

        low_bound = torch.max(quad_low_bound, simp_low_bound)
        up_bound = torch.min(quad_up_bound, simp_up_bound)

        return low_bound, up_bound, W_list, m1_list, m2_list

class TanhLayer(Layer):

    def __init__(self,):

        super(TanhLayer, self).__init__()

    def forward(self, x):

        return torch.tanh(x)

    def bound(self, l, u, W_list, m1_list, m2_list, ori_perturb_norm = None, ori_perturb_eps = None, first_layer = False):

        assert first_layer == False, 'the first layer cannot be ReLU'

        batch_size = l.shape[0]

        # quad method
        # Obtain D, m1, m2
        D, m1, m2 = linearize_tanh(l, u)
        D = D.reshape(batch_size, -1)
        m1 = m1.reshape(batch_size, -1)
        m2 = m2.reshape(batch_size, -1)
        out_dim = D.shape[1]

        # For potential merge
        max_var = torch.max(torch.abs(m1_list[-1] - m2_list[-1]))

        # Update
        update_list = W_list if max_var > 1e-4 else W_list[:-1]
        for idx, W in enumerate(update_list):
            W_list[idx] = reduced_bm_bm(D, W)

        # Add the contribution of this layer
        if max_var > 1e-4:
            W_list.append(torch.ones([batch_size, out_dim], device = D.device))
            m1_list.append(m1)
            m2_list.append(m2)
        else:
            m1_list[-1] = m1_list[-1] * D + m1
            m2_list[-1] = m2_list[-1] * D + m2

        quad_low_bound, quad_up_bound = quad_bound_calc(W_list, m1_list, m2_list, ori_perturb_norm, ori_perturb_eps)

        # simp method
        simp_low_bound = torch.tanh(l)
        simp_up_bound = torch.tanh(u)

        low_bound = torch.max(quad_low_bound, simp_low_bound)
        up_bound = torch.min(quad_up_bound, simp_up_bound)

        return low_bound, up_bound, W_list, m1_list, m2_list

class ArctanLayer(Layer):

    def __init__(self,):

        super(ArctanLayer, self).__init__()

    def forward(self, x):

        return torch.atan(x)

    def bound(self, l, u, W_list, m1_list, m2_list, ori_perturb_norm = None, ori_perturb_eps = None, first_layer = False):

        assert first_layer == False, 'the first layer cannot be ReLU'

        batch_size = l.shape[0]

        # quad method
        # Obtain D, m1, m2
        D, m1, m2 = linearize_arctan(l, u)
        D = D.reshape(batch_size, -1)
        m1 = m1.reshape(batch_size, -1)
        m2 = m2.reshape(batch_size, -1)
        out_dim = D.shape[1]

        # For potential merge
        max_var = torch.max(torch.abs(m1_list[-1] - m2_list[-1]))

        # Update
        update_list = W_list if max_var > 1e-4 else W_list[:-1]
        for idx, W in enumerate(update_list):
            W_list[idx] = reduced_bm_bm(D, W)

        # Add the contribution of this layer
        if max_var > 1e-4:
            W_list.append(torch.ones([batch_size, out_dim], device = D.device))
            m1_list.append(m1)
            m2_list.append(m2)
        else:
            m1_list[-1] = m1_list[-1] * D + m1
            m2_list[-1] = m2_list[-1] * D + m2

        quad_low_bound, quad_up_bound = quad_bound_calc(W_list, m1_list, m2_list, ori_perturb_norm, ori_perturb_eps)

        # simp method
        simp_low_bound = torch.atan(l)
        simp_up_bound = torch.atan(u)

        low_bound = torch.max(quad_low_bound, simp_low_bound)
        up_bound = torch.min(quad_up_bound, simp_up_bound)

        return low_bound, up_bound, W_list, m1_list, m2_list


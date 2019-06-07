import torch
import torch.nn as nn

import numpy as np

def reduced_m_bm(m1, m2):
    '''
    >>> merge a new constant transformation with a batch transformation

    >>> m1: tensor of shape [dim1, dim2]
    >>> m2: tensor of shape [batch_size, dim2] or [batch_size, dim2, dim3]
    '''
    assert len(m1.shape) == 2, 'The dim of m1 should be 2.'
    dim2 = len(m2.shape)

    if dim2 == 2:
        mbm = m1.unsqueeze(0) * m2.unsqueeze(1)
    elif dim2 == 3:
        mbm = torch.matmul(m1, m2)
    else:
        raise ValueError('The dim of m2 should be either 2 or 3.')

    return mbm

def reduced_bm_m(m1, m2):
    '''
    >>> merge a batch transformation with a new constant transformation

    >>> m1: tensor of shape [batch_size, dim2] or [batch_size, dim1, dim2]
    >>> m2: tensor of shape [dim2, dim3]
    '''
    assert len(m2.shape) == 2, 'The dim of m2 should be 2.'
    dim1 = len(m1.shape)

    if dim1 == 2:
        bmm = m1.unsqueeze(2) * m2.unsqueeze(0)
    elif dim1 == 3:
        bmm = torch.matmul(m1, m2)
    else:
        raise ValueError('The dim of m1 should be either 2 or 3.')

    return bmm

def reduced_bm_bm(m1, m2):
    '''
    >>> merge a batch trasformation with a new batch of transformation

    >>> m1: tensor of shape [batch_size, dim2] or [batch_size, dim1, dim2]
    >>> m2: tensor of shape [batch_size, dim2] or [batch_size, dim2, dim3]
    '''
    dim1 = len(m1.shape)
    dim2 = len(m2.shape)

    if (dim1, dim2) == (2, 2):
        bmbm = m1 * m2
    elif (dim1, dim2) == (2, 3):
        bmbm = m1.unsqueeze(2) * m2
    elif (dim1, dim2) == (3, 2):
        bmbm = m1 * m2.unsqueeze(1)
    elif (dim1, dim2) == (3, 3):
        bmbm = torch.matmul(m1, m2)
    else:
        raise ValueError('The dim of m1 and m2 should be either 2 or 3.')

    return bmbm

def reduced_bv_bm(m1, m2):
    '''
    >>> merge a batch of values with a batch of transformation

    >>> m1: tensor of shape [batch_size, dim1]
    >>> m2: tensor of shape [batch_size, dim1] or [batch_size, dim1, dim2]
    '''
    assert len(m1.shape) == 2, 'The dim of m1 should be 2.'
    dim2 = len(m2.shape)

    if dim2 == 2:
        bvbm = m1 * m2
    elif dim2 == 3:
        bvbm = torch.matmul(m1.unsqueeze(1), m1).squeeze(1)
    else:
        raise ValueError('The dim of m2 should be either 2 or 3.')

    return bvbm

def reduced_bm_bv(m1, m2):
    '''
    >>> merge a batch of transformation with a batch of values

    >>> m1: tensor of shape [batch_size, dim2] or [batch_size, dim1, dim2]
    >>> m2: tensor of shape [batch_size, dim2]
    '''
    assert len(m2.shape) == 2, 'The dim of m2 should be 2.'
    dim1 = len(m1.shape)

    if dim1 == 2:
        bmbv = m1 * m2
    elif dim1 == 3:
        bmbv = torch.matmul(m1, m2.unsqueeze(2)).squeeze(2)
    else:
        raise ValueError('The dim of m1 should be either 2 or 3.')

    return bmbv

def quad_bound_calc(W_list, m1_list, m2_list, ori_perturb_norm = None, ori_perturb_eps = None):
    '''
    >>> W_list, m1_list, m2_list: The transition matrix, lower bound input and upper bound input.
    >>> ori_perturb_norm: float, the norm of initial perturbation
    >>> ori_perturb_eps: tensor of shape [batch_size, in_dim]
    '''

    up_bound = 0.
    low_bound = 0.
    if ori_perturb_norm != None:
        primal_norm = ori_perturb_norm
        dual_norm = 1. / (1. - 1. / primal_norm)
        up_bound = torch.norm(W_list[0] * ori_perturb_eps.unsqueeze(1), dim = 2, p = dual_norm)         # of shape [batch_size, out_dim]
        low_bound = - up_bound

    for W, m1, m2 in zip(W_list, m1_list, m2_list):
        W_neg = torch.clamp(W, max = 0.)
        W_pos = torch.clamp(W, min = 0.)

        up_bound = up_bound + reduced_bm_bv(W_pos, m2) + reduced_bm_bv(W_neg, m1)
        low_bound = low_bound + reduced_bm_bv(W_pos, m1) + reduced_bm_bv(W_neg, m2)

    return low_bound, up_bound



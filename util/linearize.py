import torch
import torch.nn
import torch.nn.functional as F

# The minimum non-zero slope, to avoid numeric unstability
D_threshold = 1e-2

'''
>>> linearization function
>>> low_bound: torch.Tensor, of shape [batch_size, dim]
>>> up_bound: torch.Tensor, of shape [batch_size, dim]

return
>>> D: slope, of shape [batch_size, dim]
>>> m1: lower bound bias term, of shape [batch_size, dim]
>>> m2: upper bound bias term, of shape [batch_size, dim]
'''

def linearize_relu(low_bound, up_bound):
    '''
    slope = (relu(u) - relu(l)) / (u - l)
    m1 = 0
    m2 = relu(l) - (relu(u) - relu(l)) * l / (u - l)
    '''
    up_bound = up_bound + 1e-6
    low_bound = low_bound - 1e-6

    D = (F.relu(up_bound) - F.relu(low_bound)) / (up_bound - low_bound)
    m1 = D * 0.
    m2 = F.relu(low_bound) - low_bound * D

    return D, m1, m2

def linearize_sigd(low_bound, up_bound):
    '''
    slope = (sigma(u) - sigma(l)) / (u - l)
    check linearization part
    '''
    up_bound = up_bound + 1e-6
    low_bound = low_bound - 1e-6

    y_low = F.sigmoid(low_bound)
    y_up = F.sigmoid(up_bound)

    D = (y_up - y_low) / (up_bound - low_bound)
    t1 = -torch.log((- (2 * D - 1) + torch.sqrt(torch.clamp(1 - 4 * D, min = 1e-6)) + 1e-6)/ (2 * D + 1e-6))
    t2 = -t1

    y1 = F.sigmoid(t1) - t1 * D - 1e-6
    y2 = 1. - y1
    y = (up_bound * y_low - low_bound * y_up) / (up_bound - low_bound)

    # round small D value to zero to avoid numeric unstability
    small_D_mask = (D < D_threshold).float()
    D = D * (1. - small_D_mask)

    neg_mask = (up_bound <= 0.).float()
    pos_mask = (low_bound >= 0.).float()
    neu_mask = 1. - neg_mask - pos_mask

    m1 = (y1 * (neg_mask + neu_mask) + y * pos_mask) * (1. - small_D_mask) + small_D_mask * y_low
    m2 = (y * neg_mask + y2 * (neu_mask + pos_mask)) * (1. - small_D_mask) + small_D_mask * y_up

    return D, m1, m2

def linearize_tanh(low_bound, up_bound):
    '''
    slope = (sigma(u) - sigma(l)) / (u - l)
    check the linearization part
    '''
    up_bound = up_bound + 1e-6
    low_bound = low_bound - 1e-6

    y_low = F.tanh(low_bound)
    y_up = F.tanh(up_bound)

    D = (y_up - y_low) / (up_bound - low_bound)
    t1 = torch.log((- (D - 2) - 2 * torch.sqrt(torch.clamp(1. - D, min = 1e-6)) + 1e-6) / (D + 1e-6)) / 2.
    t2 = -t1

    y1 = F.tanh(t1) - t1 * D - 1e-6
    y2 = -y1
    y = (up_bound * y_low - low_bound * y_up) / (up_bound - low_bound)

    # round small D value to zero to avoid numeric unstability
    small_D_mask = (D < D_threshold).float()
    D = D * (1. - small_D_mask)

    neg_mask = (up_bound <= 0.).float()
    pos_mask = (low_bound >= 0.).float()
    neu_mask = 1. - neg_mask - pos_mask

    m1 = (y1 * (neg_mask + neu_mask) + y * pos_mask) * (1. - small_D_mask) + small_D_mask * y_low
    m2 = (y * neg_mask + y2 * (neu_mask + pos_mask)) * (1. - small_D_mask) + small_D_mask * y_up

    return D, m1, m2

def linearize_arctan(low_bound, up_bound):
    '''
    slope = (sigma(u) - sigma(l)) / (u - l)
    check the linearization part
    '''
    up_bound = up_bound + 1e-6
    low_bound = low_bound - 1e-6

    y_low = torch.atan(low_bound)
    y_up = torch.atan(up_bound)
    low_sign = torch.sign(low_bound)
    up_sign = torch.sign(up_bound)

    D = (y_up - y_low) / (up_bound - low_bound)
    t1 = - torch.sqrt(1. / D - 1.)
    t2 = -t1

    y1 = torch.atan(t1) - t1 * D - 1e-6
    y2 = -y1
    y = (up_bound * y_low - low_bound * y_up) / (up_bound - low_bound)

    # round small D value to zero to avoid numeric unstability
    small_D_mask = (D < D_threshold).float()
    D = D * (1. - small_D_mask)

    neg_mask = (up_bound <= 0.).float()
    pos_mask = (low_bound >= 0.).float()
    neu_mask = 1. - neg_mask - pos_mask

    m1 = (y1 * (neg_mask + neu_mask) + y * pos_mask) * (1. - small_D_mask) + small_D_mask * y_low
    m2 = (y * neg_mask + y2 * (neu_mask + pos_mask)) * (1. - small_D_mask) + small_D_mask * y_up

    return D, m1, m2


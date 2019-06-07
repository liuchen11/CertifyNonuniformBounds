import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

def imshow(image, dimshuffle = None, rescale = False, complementary = False):
    '''
    >>> plot the images after some transformation
    '''

    if dimshuffle != None:
        image = np.transpose(image, dimshuffle)

    if isinstance(rescale, bool) and rescale == True:                       # Whether or not to apply auto-rescaling
        if np.max(image) - np.min(image) < 1e-8:
            image = np.zeros_like(image)
        else:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
    if isinstance(rescale, (list, tuple)):                                  # rescale = [bias, scale]
        image = (image - rescale[0]) / rescale[1]

    if complementary == True:
        image = 1. - image

    image = np.clip(image, a_min = 0.0, a_max = 1.0)
    channel_num = image.shape[-1]

    if channel_num == 3:
        plt.imshow(image)
    elif channel_num == 1:
        stacked_image = np.concatenate([image, image, image], axis = 2)
        plt.imshow(stacked_image)
    else:
        raise ValueError('Unsupported channel num: %d'%channel_num)

    plt.xticks([])
    plt.yticks([])

def imselect(image, feature_num, dimshuffle = None, rescale = False, complementary = False):
    '''
    >>> plot the image only highlighting the top features
    '''

    # Transformation
    if dimshuffle != None:
        image = np.transpose(image, dimshuffle)

    if rescale == True:
        if np.max(image) - np.min(image) < 1e-8:
            image = np.zeros_like(image)
        else:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))

    if complementary == True:
        image = 1. - image

    image = np.clip(image, a_min = 0.0, a_max = 1.0)
    channel_num = image.shape[-1]

    # Select
    norms = np.linalg.norm(image, axis = 2).reshape(-1)
    threshold = np.sort(norms)[-feature_num]
    mask = [1. if v >= threshold else 0. for v in norms]
    image = image * np.array(mask, dtype = np.float32).reshape(image.shape[0], image.shape[1], 1)

    if channel_num == 3:
        plt.imshow(image)
    elif channel_num == 1:
        stacked_image = np.concatenate([image, image, image], axis = 2)
        plt.imshow(stacked_image)
    else:
        raise ValueError('Unsupported channel num: %d'%channel_num)

    plt.xticks([])
    plt.yticks([])

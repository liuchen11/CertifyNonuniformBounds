import pickle
import numpy as np

import torch
from torchvision import datasets, transforms

def load_pkl(pkl_file, batch_size):

    info = pickle.load(open(pkl_file, 'rb'))
    data_batch = info['data']
    label_batch = info['label']
    batch_num = data_batch.shape[0] // batch_size
    point_num = data_batch.shape[0]

    while True:
        for batch_idx in range(batch_num):
            data_this_batch = data_batch[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            label_this_batch = label_batch[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            yield torch.FloatTensor(data_this_batch), torch.LongTensor(label_this_batch)

        permute_idx = np.random.permutation(point_num)
        data_batch = data_batch[permute_idx]
        label_batch = label_batch[permute_idx]

def mnist(batch_size, batch_size_test, horizontal_flip = False, random_clip = False, normalization = None):

    if normalization == None:
        normalization = transforms.Normalize(mean = [0.5,], std = [0.5,])

    basic_transform = [transforms.ToTensor(), normalization]
    data_augumentation = []
    if horizontal_flip == True:
        data_augumentation.append(transforms.RandomHorizontalFlip())
    if random_clip == True:
        data_augumentation.append(transforms.RandomCrop(28, 4))

    train_set = datasets.MNIST(root = './data', train = True, download = True,
        transform = transforms.Compose(data_augumentation + basic_transform))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)

    test_set = datasets.MNIST(root = './data', train = False, download = True,
        transform = transforms.Compose(basic_transform))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size_test, shuffle = False, num_workers = 4, pin_memory = True)

    return train_loader, test_loader

def load_mnist(batch_size, flatten = True, dset = 'test', horizontal_flip = False,
    random_clip = False, normalization = None, subset = None):

    if subset == None:

        train_loader, test_loader = mnist(batch_size = batch_size, batch_size_test = batch_size,
            horizontal_flip = horizontal_flip, random_clip = random_clip, normalization = normalization)

        loader = {'train': train_loader, 'test': test_loader}[dset]

        while True:
            for idx, (data_batch, label_batch) in enumerate(loader, 0):
                if data_batch.shape[0] != batch_size:
                    continue
                if flatten == True:
                    data_batch = data_batch.view(data_batch.shape[0], -1)
                yield data_batch, label_batch

    else:

        train_loader, test_loader = mnist(batch_size = 1, batch_size_test = 1, horizontal_flip = horizontal_flip,
            random_clip = random_clip, normalization = normalization)

        loader = {'train': train_loader, 'test': test_loader}[dset]
        label2mapping = {label: idx for idx, label in enumerate(subset)}

        while True:
            instance_num = 0
            data_list = []
            label_list = []
            for idx, (data_batch, label_batch) in enumerate(loader, 0):

                label = int(label_batch.data.cpu().numpy()[0])
                if label in label2mapping:
                    labelmap = label2mapping[label]
                    data_list.append(data_batch)
                    label_list.append(labelmap)

                    if len(label_list) == batch_size:
                        data_batch_cat = torch.cat(data_list, dim = 0)
                        label_batch_cat = torch.LongTensor(label_list)
                        data_list = []
                        label_list = []
                        yield data_batch_cat, label_batch_cat

def fmnist(batch_size, batch_size_test, horizontal_flip = False, random_clip = False, normalization = None):

    if normalization == None:
        normalization = transforms.Normalize(mean = [0.5,], std = [0.5,])

    basic_transform = [transforms.ToTensor(), normalization]
    data_augumentation = []
    if horizontal_flip == True:
        data_augumentation.append(transforms.RandomHorizontalFlip())
    if random_clip == True:
        data_augumentation.append(transforms.RandomCrop(28, 4))

    train_set = datasets.FashionMNIST(root = './data', train = True, download = True,
        transform = transforms.Compose(data_augumentation + basic_transform))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)

    test_set = datasets.FashionMNIST(root = './data', train = False, download = True,
        transform = transforms.Compose(basic_transform))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size_test, shuffle = False, num_workers = 4, pin_memory = True)

    return train_loader, test_loader

def load_fmnist(batch_size, flatten = True, dset = 'test', horizontal_flip = False,
    random_clip = False, normalization = None, subset = None):

    if subset == None:

        train_loader, test_loader = fmnist(batch_size = batch_size, batch_size_test = batch_size,
            horizontal_flip = horizontal_flip, random_clip = random_clip, normalization = normalization)

        loader = {'train': train_loader, 'test': test_loader}[dset]

        while True:
            for idx, (data_batch, label_batch) in enumerate(loader, 0):
                if data_batch.shape[0] != batch_size:
                    continue
                if flatten == True:
                    data_batch = data_batch.view(data_batch.shape[0], -1)
                yield data_batch, label_batch
    else:
        raise NotImplementedError('This part has not been implemented yet.')

def svhn(batch_size, batch_size_test, horizontal_flip = False, random_clip = False, normalization = None):

    if normalization == None:
        normalization = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])

    basic_transform = [transforms.ToTensor(), normalization]
    data_augumentation = []

    if horizontal_flip == True:
        data_augumentation.append(transforms.RandomHorizontalFlip())
    if random_clip == True:
        data_augumentation.append(transforms.RandomCrop(28, 4))

    train_set = datasets.SVHN(root = './data', split = 'train', download = True,
        transform = transforms.Compose(data_augumentation + basic_transform))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)

    test_set = datasets.SVHN(root = './data', split = 'test', download = True,
        transform = transforms.Compose(data_augumentation + basic_transform))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size_test, shuffle = True, num_workers = 4, pin_memory = True)

    return train_loader, test_loader

def load_svhn(batch_size, flatten = True, dset = 'test', horizontal_flip = False,
    random_clip = False, normalization = None, subset = None):

    if subset == None:

        train_loader, test_loader = svhn(batch_size = batch_size, batch_size_test = batch_size,
            horizontal_flip = horizontal_flip, random_clip = random_clip, normalization = normalization)

        loader = {'train': train_loader, 'test': test_loader}[dset]

        while True:
            for idx, (data_batch, label_batch) in enumerate(loader, 0):
                if data_batch.shape[0] != batch_size:
                    continue
                if flatten == True:
                    data_batch = data_batch.view(data_batch.shape[0], -1)
                yield data_batch, label_batch
    else:
        raise NotImplementedError('This part has been implemented yet.')


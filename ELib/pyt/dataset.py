'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: dataset.py
@time: 2019-10-18 10:14:15
@desc: 
'''
import torch.utils.data as data
import pickle
import numpy as np
import random
import PIL.Image as Image


class BasicDataSet(object):
    def __init__(self, root, train_ratio=1):
        self._root_path = root
        self.ratio=train_ratio

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def read(self, onehot=False, channel_first=True):
        (x_train, targets_train), (x_test, targets_test) = self._readData(channel_first)

        x_total = np.concatenate((x_train, x_test))
        y_total = np.concatenate((targets_train, targets_test))

        index_list = list(range(0, x_total.shape[0]))
        random.shuffle(index_list)

        train_record_count = int(len(index_list) * self.ratio) if self.ratio > 0 else len(x_train)

        index_train = index_list[0:train_record_count]
        index_test  = index_list[train_record_count:len(index_list)]

        x_train = x_total[index_train]
        x_test = x_total[index_test]
        targets_train = y_total[index_train]
        targets_test = y_total[index_test]

        self.TRAIN_RECORDS = x_train.shape[0]
        self.TEST_RECORDS = x_test.shape[0]

        if onehot:
            y_train = np.zeros((targets_train.shape[0], 10), dtype = np.uint8)
            y_test = np.zeros((targets_test.shape[0], 10), dtype = np.uint8)
            y_train[np.arange(targets_train.shape[0]), targets_train] = 1
            y_test[np.arange(targets_test.shape[0]), targets_test] = 1

            return (x_train, y_train), (x_test, y_test)
        else:
            return (x_train, np.reshape(targets_train, newshape=(targets_train.shape[0], 1))), (x_test, np.reshape(targets_test, newshape=(targets_test.shape[0], 1)))


    def _readData(self, channel_first):
        pass


class MnistDataSet(BasicDataSet):
    NUM_OUTPUTS = 10
    IMAGE_SIZE = 28
    IMAGE_CHANNEL = 1

    def __init__(self, root="/data/input/mnist.npz", radio=1):
        super(MnistDataSet, self).__init__(root=root, train_ratio=radio)

    def _readData(self, channel_first):
        f = np.load(self._root_path)
        x_train, targets_train = f['x_train'], f['y_train']
        x_test, targets_test = f['x_test'], f['y_test']
        f.close()

        return (x_train, targets_train), (x_test, targets_test)


class MnistDatasetForPytorch(data.Dataset):
    def __init__(self, root="/data/input/mnist.npz", train=True, radio=0.9, one_hot=False, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        reader = MnistDataSet(root=self.root, radio=radio)

        (self.train_data, self.train_labels),(self.test_data, self.test_labels) = reader.read(onehot=one_hot)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

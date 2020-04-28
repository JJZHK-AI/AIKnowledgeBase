'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: kensor.py
@time: 2019-07-15 14:42:37
@desc: 
'''
import collections
import math
import functools as ft


class KTensor(object):
    def __init__(self, x, grad=None):
        self.data = x
        self.grad = grad
        self.shape = x.shape

    def tonumpy(self):
        return self.data

    def backward(self):
        pass


class KModule(object):
    def __init__(self):
        self.shape = None
        self._modules = collections.OrderedDict()
        self._forward_hook = collections.OrderedDict()
        self._backward_hook = collections.OrderedDict()

    def __call__(self, image=KTensor, *args, **kwargs): # forward
        if self.shape is None:
            self._initial_paramters_(image)

        return self.forward(image, *args)

    def _initial_paramters_(self, x=KTensor):
        self.shape = x.shape
        self.init_paramters(x)

    def add_modules(self, name, layer):
        self._modules[name] = layer

    def init_paramters(self, x=KTensor):
        pass

    def forward(self, x=KTensor, *args):
        pass


class KSequential(KModule):
    def __init__(self, *args):
        super(KSequential, self).__init__()
        for idx, layer in enumerate(args):
            self.add_modules(str(idx), layer)

    def init_paramters(self, x=KTensor):
        pass

    def forward(self, x=KTensor, *args):
        tempImage = x
        for layer in self._modules.values():
            if layer.shape is None:
                layer._initial_paramters_(tempImage)

            tempImage = layer.forward(tempImage)

        return KTensor(tempImage.data, grad=self._backward_hook)


class Conv2D(KModule):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.padding = padding

    def init_paramters(self, x=KTensor):
        self.batchsize = self.shape[0]

        weights_scale = math.sqrt(ft.reduce(lambda x, y: x * y, self.shape) / self.out_channels)
        self.weights = np.random.standard_normal(
            (self.ksize, self.ksize, self.in_channels, self.out_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.out_channels) / weights_scale

        if self.padding == 1:
            self.eta = np.zeros(shape=(self.batchsize,
                                       int((self.shape[1] - self.ksize + 1) / self.stride),
                                       int((self.shape[2] - self.ksize + 1) / self.stride),
                                       self.out_channels))
        elif self.padding == 0:
            self.eta = np.zeros(shape=(self.batchsize,
                                       int(self.shape[1] / self.stride),
                                       int(self.shape[2] / self.stride),
                                       self.out_channels))

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

        if (self.shape[1] - self.ksize) % self.stride != 0:
            raise Exception('input tensor width can\'t fit stride')
        if (self.shape[2] - self.ksize) % self.stride != 0:
            raise Exception('input tensor height can\'t fit stride')

    def forward(self, x=KTensor, *args):
        col_weights = self.weights.reshape([-1, self.out_channels])
        tempData = x.tonumpy()
        if self.padding == 0:
            tempData = np.pad(tempData, (
                (0, 0), (int(self.ksize / 2), int(self.ksize / 2)), (int(self.ksize / 2), int(self.ksize / 2)), (0, 0)),
                       'constant', constant_values=0)

        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = tempData[i][np.newaxis, :]
            self.col_image_i = self._im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return KTensor(conv_out)

    def _im2col(self, image, ksize, stride):
        image_col = []
        for i in range(0, image.shape[1] - ksize + 1, stride):
            for j in range(0, image.shape[2] - ksize + 1, stride):
                col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
                image_col.append(col)
        image_col = np.array(image_col)

        return image_col

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import ELib.pyn.cnn.functions as epcf

    plt.figure(figsize=(10, 10), facecolor='w')
    imagePath = "../../../Results/03/01/ConvVisible01.jpg"
    img = cv2.imread(imagePath)
    basicImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    originImg = cv2.resize(basicImage, (224, 224))
    plt.subplot(1, 2, 1)
    plt.title('Origin')
    plt.imshow(originImg)

    model = KSequential(
        Conv2D(3, 64, 3, 1, 1)
    )
    img = epcf.preprocess_image(basicImage).transpose((0, 2, 3, 1))
    img = KTensor(img)
    features = model(img).tonumpy()
    feature = features[:, :, :, 0]
    feature = np.reshape(feature, newshape=(feature.shape[1], feature.shape[2]))
    feature = 1.0 / (1 + np.exp(-1 * feature))
    feature = np.round(feature * 255)
    plt.subplot(1, 2, 2)
    plt.title('Conv2D')
    plt.imshow(feature, cmap='gray')

    plt.show()
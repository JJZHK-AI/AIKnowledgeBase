import math
import functools as ft
import numpy as np

class Op(object):
    def __init__(self):
        self.shape = None

    def forward(self, x, *args, **kwargs):
        pass

    def init_paramters(self, x):
        pass

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        pass

    def gradient(self, eta):
        pass

    def _init_paramters_(self, x):
        self.shape = x.shape
        self.init_paramters(x)

    def __call__(self, x, *args, **kwargs):
        if self.shape == None:
            self._init_paramters_(x)

        return self.forward(x, *args, **kwargs)


class Conv2D(Op):
    def __init__(self, input_channel, output_channel, ksize=3, stride=1, method='VALID'):
        super(Conv2D, self).__init__()
        self.output_channel = output_channel
        self.input_channel = input_channel
        self.ksize = ksize
        self.stride = stride
        self.method = method

    def forward(self, x, *args, **kwargs):
        col_weights = self.weights.reshape([-1, self.output_channel])
        if self.method == 'SAME':
            x = np.pad(x, (
                (0, 0), (int(self.ksize / 2), int(self.ksize / 2)), (int(self.ksize / 2), int(self.ksize / 2)), (0, 0)),
                       'constant', constant_values=0)

        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = self._im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def init_paramters(self, x):
        self.batchsize = self.shape[0]

        weights_scale = math.sqrt(ft.reduce(lambda x, y : x * y, self.shape) / self.output_channel)
        self.weights = np.random.standard_normal((self.ksize, self.ksize, self.input_channel, self.output_channel)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channel) / weights_scale

        if self.method == 'VALID':
            self.eta = np.zeros(shape=(self.batchsize,
                                       int((self.shape[1] - self.ksize + 1) / self.stride),
                                       int((self.shape[2] - self.ksize + 1) / self.stride),
                                       self.output_channel))
        elif self.method == 'SAME':
            self.eta = np.zeros(shape=(self.batchsize,
                                       int(self.shape[1] / self.stride),
                                       int(self.shape[2] / self.stride),
                                       self.output_channel))

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

        if (self.shape[1] - self.ksize) % self.stride != 0:
            raise Exception('input tensor width can\'t fit stride')
        if (self.shape[2] - self.ksize) % self.stride != 0:
            raise Exception('input tensor height can\'t fit stride')

    def gradient(self, eta):
        self.eta = eta
        col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channel])

        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_eta
        if self.method == 'VALID':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.method == 'SAME':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                             'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channel])
        col_pad_eta = np.array([self._im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.shape)
        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def _im2col(self, image, ksize, stride):
        # image is a 4d tensor([batchsize, width ,height, channel])
        image_col = []
        for i in range(0, image.shape[1] - ksize + 1, stride):
            for j in range(0, image.shape[2] - ksize + 1, stride):
                col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
                image_col.append(col)
        image_col = np.array(image_col)

        return image_col


class ReLU(Op):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x, *args, **kwargs):
        self.x = x
        return np.maximum(x, 0)

    def init_paramters(self, x):
        self.eta = np.zeros(self.shape)
        self.output_shape = self.shape

    def gradient(self, eta):
        self.eta = eta
        self.eta[self.x<0]=0
        return self.eta

class MaxPooling2D(Op):
    def __init__(self, ksize=2, stride=2):
        super(MaxPooling2D, self).__init__()
        self.ksize = ksize
        self.stride = stride

    def forward(self, x, *args, **kwargs):
        out = np.zeros([x.shape[0], int(x.shape[1] / self.stride), int(x.shape[2] / self.stride), self.output_channels])

        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        out[b, int(i / self.stride), int(j / self.stride), c] = np.max(
                            x[b, i:i + self.ksize, j:j + self.ksize, c])
                        index = np.argmax(x[b, i:i + self.ksize, j:j + self.ksize, c])
                        self.index[b, i + int(index / self.stride), j + index % self.stride, c] = 1
        return out

    def init_paramters(self, x):
        self.output_channels = self.shape[-1]
        self.index = np.zeros(self.shape)
        self.output_shape = [self.shape[0], int(self.shape[1] / self.stride), int(self.shape[2] / self.stride), self.output_channels]

    def gradient(self, eta):
        return np.repeat(np.repeat(eta, self.stride, axis=1), self.stride, axis=2) * self.index

class AvgPooling2D(Op):
    def __init__(self, ksize=2, stride=2):
        super(AvgPooling2D, self).__init__()
        self.ksize = ksize
        self.stride = stride

    def forward(self, x, *args, **kwargs):
        out = np.zeros([x.shape[0], int(x.shape[1] / self.stride), int(x.shape[2] / self.stride), self.output_channels])

        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        out[b, int(i / self.stride), int(j / self.stride) , c] = np.mean(
                            x[b, i:i + self.ksize, j:j + self.ksize, c])

        return out

    def init_paramters(self, x):
        self.output_channels = self.shape[-1]
        self.integral = np.zeros(self.shape)
        self.index = np.zeros(self.shape)


class FullConnect(Op):
    def __init__(self, output_channel=2):
        super(FullConnect, self).__init__()
        self.output_channel = output_channel
        self.x = None

    def forward(self, x, *args, **kwargs):
        self.x = x.reshape([self.batchsize, -1])
        output = np.dot(self.x, self.weights) + self.bias
        return output

    def init_paramters(self, x):
        self.batchsize = self.shape[0]

        input_len = ft.reduce(lambda x, y: x * y, self.shape[1:])

        self.weights = np.random.standard_normal((input_len, self.output_channel)) / 100
        self.bias = np.random.standard_normal(self.output_channel) / 100

        self.output_shape = [self.batchsize, self.output_channel]
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def gradient(self, eta):
        for i in range(eta.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            eta_i = eta[i][:, np.newaxis].T
            self.w_gradient += np.dot(col_x, eta_i)
            self.b_gradient += eta_i.reshape(self.bias.shape)

        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.shape)

        return next_eta

class Softmax(object):
    def __init__(self):
        self.shape = None

    def __call__(self, x, *args, **kwargs):
        if self.shape == None:
            self.init_paramters(x)

        self.label = args[0]
        self.prediction = x
        self._predict_(self.prediction)
        self.loss = 0
        for i in range(self.batchsize):
            self.loss += np.log(np.sum(np.exp(self.prediction[i]))) - self.prediction[i, self.label[i]]

        return self.loss, self.softmax

    def init_paramters(self, x):
        self.shape = x.shape
        self.softmax = np.zeros(self.shape)
        self.eta = np.zeros(self.shape)
        self.batchsize = self.shape[0]

    def _predict_(self, prediction):
        exp_prediction = np.zeros(prediction.shape)
        self.softmax = np.zeros(prediction.shape)
        for i in range(self.batchsize):
            prediction[i, :] -= np.max(prediction[i, :])
            exp_prediction[i] = np.exp(prediction[i])
            self.softmax[i] = exp_prediction[i]/np.sum(exp_prediction[i])
        return self.softmax


    def gradient(self):
        self.eta = self.softmax.copy()
        for i in range(self.batchsize):
            self.eta[i, self.label[i]] -= 1
        return self.eta

# if __name__ == '__main__':
#     import cv2
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     import torchvision.models as models
#     import torch
#     import functions as epcf
#
#     plt.figure(figsize=(10, 10), facecolor='w')
#     imagePath = "../../../data/ConvVisible01.jpg"
#     img = cv2.imread(imagePath)
#     basicImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     originImg = cv2.resize(basicImage, (224, 224))
#     plt.subplot(1, 3, 1)
#     plt.title('Origin')
#     plt.imshow(originImg)
#
#     conv1 = Conv2D(3, 64, 3, 1, 'SAME')
#     img = epcf.preprocess_image(basicImage).transpose((0, 2, 3, 1))
#     features = conv1(img)
#     feature = features[:, :, :, 0]
#     feature = np.reshape(feature, newshape=(feature.shape[1], feature.shape[2]))
#     feature = 1.0 / (1 + np.exp(-1 * feature))
#     feature = np.round(feature * 255)
#     plt.subplot(1, 3, 2)
#     plt.title('Conv2D')
#     plt.imshow(feature, cmap='gray')
#
#     model = models.vgg16(pretrained=True).features
#     img = epcf.preprocess_image(basicImage)
#     features = model[0](torch.autograd.Variable(torch.FloatTensor(img)))
#     feature = features[:, 0, :, :]
#     feature = feature.view(feature.shape[1], feature.shape[2])
#     feature = feature.data.numpy()
#     feature = 1.0 / (1 + np.exp(-1 * feature))
#     feature = np.round(feature * 255)
#     plt.subplot(1, 3, 3)
#     plt.title('Pytorch Conv2D')
#     plt.imshow(feature, cmap='gray')
#     plt.show()
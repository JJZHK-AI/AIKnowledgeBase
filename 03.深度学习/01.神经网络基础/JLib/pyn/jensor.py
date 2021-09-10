import numpy as np
import functools as ft
import math


if 'GLOBAL_VARIABLE_SCOPE' not in globals():
    GLOBAL_VARIABLE_SCOPE = {}


class Jensor(object):
    def __init__(self, name):
        self.name = name

        self.prev = []
        self.next = []

class Variable(Jensor):
    initial = 'MSRA'
    method = 'SGD'
    def __init__(self, shape=list, name=str, scope='', grad=True, learnable=False):
        if scope != '':
            self.scope = scope if scope[-1] == '/' else scope + '/'
            self.name = self.scope + name
        else:
            self.name = name
            self.scope = scope
        super(Variable, self).__init__(self.name)

        if self.name in GLOBAL_VARIABLE_SCOPE:
            raise Exception('Variable name: %s exists!' % self.name)
        else:
            GLOBAL_VARIABLE_SCOPE[self.name] = self

        for i in shape:
            if not isinstance(i, int):
                raise Exception("Variable name: %s shape is not list of int"%self.name)

        self.shape = shape
        self.data = self._initializer(shape, self.initial)

        if grad:
            self.diff = np.zeros(self.shape)
            self.wait_bp = True
            self.learnable = learnable

    def _initializer(self, shape, method):
        if method == 'const':
            return np.random.standard_normal(shape) / 100

        if method == 'None':
            return np.zeros(shape)

        if method == 'MSRA':
            weights_scale = math.sqrt(ft.reduce(lambda x, y: x * y, shape) / shape[-1])
            return np.random.standard_normal(shape) / weights_scale

    def eval(self):
        for operator in self.prev:
            GLOBAL_VARIABLE_SCOPE[operator].forward()
        self.wait_bp = True
        return self.data

    def diff_eval(self):
        if self.wait_bp:
            for operator in self.next:
                GLOBAL_VARIABLE_SCOPE[operator].backward()
            self.wait_bp = False
        else:
            pass

        return self.diff

    def apply_gradient(self, learning_rate=float, decay_rate=float, batch_size=1):
        self.data *= (1 - decay_rate)
        if self.method == 'SGD':
            learning_rate = learning_rate
            self.data -= (learning_rate*self.diff/batch_size)
            self.diff *= 0

        elif self.method == 'Momentum':
            self.mtmp = self.momentum * self.mtmp + self.diff/batch_size
            self.data -= learning_rate * self.mtmp
            self.diff *= 0

        elif self.method == 'NGA':
            self.mtmp = self.momentum * self.mtmp + self.diff / batch_size + self.momentum*(self.diff-self.lastdiff)/batch_size
            self.data -= learning_rate * self.mtmp
            self.lastdiff = self.diff
            self.diff *= 0

        elif self.method == 'Adam':
            self.t += 1
            learning_rate_t = learning_rate * math.sqrt(1 - pow(self.beta2, self.t)) / (1 - pow(self.beta1, self.t))
            self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * self.diff / batch_size
            self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * ((self.diff / batch_size) ** 2)
            self.data -= learning_rate_t * self.m_t / (self.v_t + self.epsilon) ** 0.5
            self.diff *= 0

        else:
            raise Exception('No apply_gradient method: %s'%self.method)

    def set_method_sgd(self):
        self.method = 'SGD'

    def set_method_momentum(self, momentum=0.9):
        self.method = 'Momentum'
        self.momentum = momentum
        self.mtmp = np.zeros(self.diff.shape)

    def set_method_nga(self,momentum=0.9):
        self.method = 'NGA'
        self.lastdiff = np.zeros(self.diff.shape)
        self.momentum= momentum
        self.mtmp = np.zeros(self.diff.shape)

    def set_method_adam(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.method = 'Adam'
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_t = np.zeros(self.diff.shape)
        self.v_t = np.zeros(self.diff.shape)
        self.t = 0


class Operator(Jensor):
    def __init__(self, name, input_variables, output_variables):
        super(Operator, self).__init__(name)

        if name in GLOBAL_VARIABLE_SCOPE.keys():
            raise Exception("Operator %s has exists !"%name)

        if not isinstance(input_variables, Variable) and not isinstance(input_variables[0], Variable):
            raise Exception("Operator %s 's input_variables is not instance(or list) of Variable!")

        if not isinstance(output_variables, Variable) and not isinstance(output_variables[0], Variable):
            raise Exception("Operator %s 's output_variables is not instance(or list) of Variable!")

        GLOBAL_VARIABLE_SCOPE[self.name] = self

        # register for input Variable's child and output Variable's parents
        self._register_graph(input_variables, output_variables, self)

        self.wait_forward = True
        # self.wait_backward = not self.wait_forward

    def forward(self):
        pass

    def backward(self):
        pass

    def _register_graph(self, input_variable, output_variable, operator):
        if isinstance(input_variable,Variable) and isinstance(output_variable, Variable):
            input_variable.next.append(operator.name)
            output_variable.prev.append(operator.name)
            operator.prev.append(input_variable.name)
            operator.next.append(output_variable.name)

        elif isinstance(input_variable, Variable) and len(output_variable)>1:
            for output in output_variable:
                input_variable.next.append(operator.name)
                output.prev.append(operator.name)
                operator.prev.append(input_variable.name)
                operator.next.append(output.name)

        elif isinstance(output_variable, Variable) and len(input_variable)>1:
            for _input in input_variable:
                _input.next.append(operator.name)
                output_variable.prev.append(operator.name)
                operator.prev.append(_input.name)
                operator.next.append(output_variable.name)

        elif len(output_variable)> 1 and len(input_variable)> 1:
            for _input in input_variable:
                _input.next.append(operator.name)
                operator.prev.append(_input.name)
            for output in output_variable:
                output.prev.append(operator.name)
                operator.next.append(output.name)

        else:
            raise Exception('Operator name %s input,output list error'% operator.name)


class Conv2D(Operator):
    def __init__(self, kernel_shape=list, input_variable=Variable, name=str, stride=1, padding='SAME'):
        # kernel_shape = [ksize, ksize, input_channels, output_channels]
        for i in kernel_shape:
            if not isinstance(i, int):
                raise Exception("Operator Conv2D name: %s kernel shape is not list of int" % self.name)

        if not isinstance(input_variable, Variable):
            raise Exception("Operator Conv2D name: %s's input_variable is not instance of Variable" % name)

        if len(input_variable.shape)!=4:
            raise Exception("Operator Conv2D name: %s's input_variable's shape != 4d Variable!" % name)

        self.ksize = kernel_shape[0]
        self.stride = stride
        self.output_num = kernel_shape[-1]
        self.padding = padding
        self.col_image = []

        self.weights = Variable(kernel_shape, scope=name, name='weights',learnable=True)
        self.bias    = Variable([self.output_num], scope=name, name='bias', learnable=True)
        self.batch_size = input_variable.shape[0]

        if self.padding == 'SAME':
            _output_shape = [self.batch_size, int(input_variable.shape[1] / stride), int(input_variable.shape[2] / stride),
                             self.output_num]
        if self.padding == 'VALID':
            _output_shape = [self.batch_size, int((input_variable.shape[1] - self.ksize + 1) / stride),
                             int((input_variable.shape[2] - self.ksize + 1) / stride), self.output_num]

        self.output_variables = Variable(_output_shape, name='out', scope=name)  # .name
        self.input_variables = input_variable
        super(Conv2D, self).__init__(name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.prev:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self._conv(self.input_variables, self.output_variables, self.weights.data, self.bias.data)
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.next:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self._deconv(self.input_variables, self.output_variables, self.weights, self.bias)
            self.wait_forward = True
            return

    def _deconv(self, input=Variable, output=Variable, weights=Variable, bias=Variable):
        col_eta = np.reshape(output.diff, [self.batch_size, -1, self.output_num])
        for i in range(self.batch_size):
            weights.diff += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        bias.diff += np.sum(col_eta, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_eta
        if self.padding == 'VALID':
            pad_eta = np.pad(output.diff, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.padding == 'SAME':
            pad_eta = np.pad(output.diff, (
                (0, 0), (int(self.ksize / 2), int(self.ksize / 2)), (int(self.ksize / 2), int(self.ksize / 2)), (0, 0)),
                             'constant', constant_values=0)

        col_pad_eta = np.array([self._im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batch_size)])
        flip_weights = np.flipud(np.fliplr(weights.data))
        flip_weights = flip_weights.swapaxes(2,3)
        col_flip_weights = flip_weights.reshape([-1, weights.shape[2]])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, input.shape)
        input.diff = next_eta
        return

    def _conv(self, input=Variable, output=Variable, weights=np.ndarray, bias=np.ndarray):
        # reshape weights to col
        col_weights = weights.reshape(-1, self.output_num)

        # padding input_img according to method
        if self.padding == 'SAME':
            batch_img = np.pad(input.data, (
                (0, 0), (int(self.ksize / 2), int(self.ksize / 2)), (int(self.ksize / 2), int(self.ksize / 2)), (0, 0)),
                               'constant', constant_values=0)
        else:
            batch_img = input.data

        # malloc tmp output_data
        conv_out = np.zeros(output.data.shape)

        self.col_image = []
        # do dot for every image in batch by im2col dot col_weight
        for i in range(self.batch_size):
            img_i = batch_img[i][np.newaxis, :]
            col_image_i = self._im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(col_image_i, col_weights) + bias, output.data[0].shape)
            self.col_image.append(col_image_i)
        self.col_image = np.array(self.col_image)

        output.data = conv_out
        return

    def _im2col(self, image, ksize, stride):
        # image is a 4d tensor([batchsize, width ,height, channel])
        # print image.shape
        # print image
        image_col = []
        for i in range(0, image.shape[1] - ksize + 1, stride):
            for j in range(0, image.shape[2] - ksize + 1, stride):
                col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])

                image_col.append(col)
        image_col = np.array(image_col)

        return image_col


class MaxPooling(Operator):
    def __init__(self, ksize=2, input_variable=Variable, name=str, stride=2):

        if not isinstance(input_variable, Variable):
            raise Exception("Operator Conv2D name: %s's input_variable is not instance of Variable" % name)


        self.ksize = ksize
        self.stride = stride
        self.batch_size = input_variable.shape[0]
        self.output_channels = input_variable.shape[-1]
        self.index = np.zeros(input_variable.shape)

        self.input_variables = input_variable
        _output_shape = [self.batch_size, int(input_variable.shape[2] / stride), int(input_variable.shape[2] / stride),
                         self.output_channels]
        self.output_variables = Variable(_output_shape, name='out', scope=name)
        super(MaxPooling, self).__init__(name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.prev:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self._pool()
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.next:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables.diff = np.repeat(np.repeat(self.output_variables.diff, self.stride, axis=1),
                                                  self.stride, axis=2) * self.index
            self.wait_forward = True
            return

    def _pool(self):
        _out = np.zeros(self.output_variables.shape)
        for b in range(self.input_variables.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, self.input_variables.shape[1], self.stride):
                    for j in range(0, self.input_variables.shape[2], self.stride):
                        _out[b, int(i / self.stride), int(j / self.stride), c] = np.max(
                            self.input_variables.data[b, i:i + self.ksize, j:j + self.ksize, c])
                        index = np.argmax(self.input_variables.data[b, i:i + self.ksize, j:j + self.ksize, c])
                        self.index[b, i+int(index/self.stride), j + index % self.stride, c] = 1
        self.output_variables.data = _out
        return


class FullyConnect(Operator):
    def __init__(self, output_num, input_variable=Variable, name=str):
        if not isinstance(input_variable, Variable):
            raise Exception("Operator Conv2D name: %s's input_variable is not instance of Variable" % name)

        self.batch_size = input_variable.shape[0]
        input_len = ft.reduce(lambda x, y: x * y, input_variable.shape[1:])
        self.output_num = output_num
        self.weights = Variable([input_len, self.output_num], name='weights', scope=name,learnable=True)
        self.bias = Variable([self.output_num], name='bias', scope=name,learnable=True)

        self.output_variables = Variable([self.batch_size, self.output_num], name='out', scope=name)
        self.input_variables = input_variable
        super(FullyConnect, self).__init__(name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:

            for parent in self.prev:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.flatten_x = self.input_variables.data.reshape([self.batch_size, -1])
            self.output_variables.data = np.dot(self.flatten_x, self.weights.data)+self.bias.data
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.next:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()

            for i in range(self.batch_size):
                col_x = self.flatten_x[i][:, np.newaxis]
                diff_i = self.output_variables.diff[i][:, np.newaxis].T
                self.weights.diff += np.dot(col_x, diff_i)
                self.bias.diff += diff_i.reshape(self.bias.shape)
            next_diff = np.dot(self.output_variables.diff, self.weights.data.T)
            self.input_variables.diff = np.reshape(next_diff, self.input_variables.shape)

            self.wait_forward = True
            return


class SoftmaxLoss(Operator):
    def __init__(self, predict = Variable, label=Variable, name=str):
        self.batch_size = predict.shape[0]
        self.input_variables = [predict, label]
        self.loss = Variable([1], name='loss', scope=name)
        self.prediction = Variable(predict.shape, name='prediction', scope=name)
        self.softmax = np.zeros(self.prediction.shape)

        self.output_variables = [self.loss, self.prediction]
        super(SoftmaxLoss, self).__init__(name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.prev:
                GLOBAL_VARIABLE_SCOPE[parent].eval()

            predict = self.input_variables[0].data
            label = self.input_variables[1].data

            self.prediction.data = self.predict(predict)

            self.loss.data = 0
            for i in range(self.batch_size):
                self.loss.data += np.log(np.sum(np.exp(predict[i]))) - predict[i, label[i]]

            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.next:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables[0].diff = self.softmax.copy()
            for i in range(self.batch_size):
                self.input_variables[0].diff[i, self.input_variables[1].data[i]] -= 1
            self.wait_forward = True
            return

    def predict(self, prediction):
        exp_prediction = np.zeros(prediction.shape)
        self.softmax = np.zeros(prediction.shape)
        for i in range(self.batch_size):
            prediction[i, :] -= np.max(prediction[i, :])
            exp_prediction[i] = np.exp(prediction[i])
            self.softmax[i] = exp_prediction[i]/np.sum(exp_prediction[i])
        return self.softmax


class DropOut(Operator):
    def __init__(self, name, phase, input_variable=Variable, prob=0.5):
        self.input_variables = input_variable
        self.output_variables = Variable(shape=input_variable.shape, scope=name, name='out')
        self.prob = prob
        self.phase = phase
        self.index = np.ones(input_variable.shape)

        super(DropOut, self).__init__(name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.prev:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            if self.phase == 'train':
                self.index = np.random.random(self.input_variables.shape) < self.prob
                self.output_variables.data = self.input_variables.data * self.index
                self.output_variables.data /= self.prob
            elif self.phase == 'test':
                self.output_variables.data = self.input_variables.data
            else:
                raise Exception('Operator %s phase is not in test or train'% self.name)

            self.wait_forward=False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.next:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            if self.phase == 'train':
                self.input_variables.diff = self.output_variables.diff * self.index / self.prob
            elif self.phase == 'test':
                self.output_variables.diff = self.input_variables.diff
            else:
                raise Exception('Operator %s phase is not in test or train'% self.name)

            self.wait_forward = True
            return


class Relu(Operator):
    def __init__(self, input_variable=Variable, name=str):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        super(Relu, self).__init__(name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.prev:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.output_variables.data = np.maximum(self.input_variables.data, 0)
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.next:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables.diff = self.output_variables.diff
            self.output_variables.diff[self.input_variables.data < 0] = 0
            self.wait_forward = True
            return


class LRelu(Operator):
    def __init__(self, input_variable=Variable, name=str, alpha = 0.01):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        self.alpha = alpha
        super(LRelu, self).__init__(name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.prev:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.output_variables.data = np.maximum(self.input_variables.data, 0) + self.alpha * np.minimum(
                self.input_variables.data, 0)
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.next:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables.diff = self.output_variables.diff
            self.input_variables.diff[self.input_variables.data <= 0] *= self.alpha
            self.wait_forward = True

            return


class Sigmoid(Operator):
    def __init__(self, input_variable=Variable, name=str):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        super(Sigmoid, self).__init__(name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.prev:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            # y = 1/(1+exp(-x))
            # print 'fuck:',self.input_variables.data
            self.output_variables.data = 1.0/(1.0+np.exp(-self.input_variables.data))
            # print 'fuck out:', self.output_variables.data
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.next:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            # eta_x = eta_y * (1-y) * y
            self.input_variables.diff = self.output_variables.data * (
                    1 - self.output_variables.data) * self.output_variables.diff
            self.wait_forward = True
            return


class Tanh(Operator):
    def __init__(self, input_variable=Variable, name=str):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        super(Tanh,self).__init__(name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.prev:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.output_variables.data = 2 * 1.0/(1.0+np.exp(-2*self.input_variables.data)) - 1
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.next:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables.diff = self.output_variables.diff * (1 - self.output_variables.data**2)
            self.wait_forward = True
            return


class Elu(Operator):
    def __init__(self, input_variable=Variable, name=str, alpha = 0.1):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        self.alpha = alpha
        super(Elu, self).__init__(name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.prev:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.output_variables.data = np.maximum(self.input_variables.data, 0) + self.alpha * (
                    np.exp(np.minimum(self.input_variables.data, 0)) - 1)
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.next:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables.diff = self.output_variables.diff
            self.output_variables.diff[self.input_variables.data <= 0] *= (
                    self.alpha * np.exp(self.input_variables.data[self.input_variables.data <= 0]))
            self.wait_forward = True
            return


class Prelu(Operator):
    def __init__(self, input_variable=Variable, name=str, alpha = 0.25):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        self.alpha = alpha
        self.momentum  = 0.9
        self.eta = 1e-4
        super(Prelu, self).__init__(name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.prev:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.output_variables.data = np.maximum(self.input_variables.data, 0) + self.alpha * np.minimum(
                self.input_variables.data, 0)
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.next:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.alpha = self.momentum * self.alpha + self.eta * np.sum(np.minimum(self.input_variables.data, 0))
            self.input_variables.diff = self.output_variables.diff
            self.input_variables.diff[self.input_variables.data <= 0] *= self.alpha
            self.wait_forward = True

            return
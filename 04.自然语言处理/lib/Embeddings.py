import torch
import math
import numpy as np

# 定义Embeddings类来实现文本嵌入层，这里s说明代表两个一模一样的嵌入层, 他们共享参数.
# 该类继承nn.Module, 这样就有标准层的一些功能, 这里我们也可以理解为一种模式, 我们自己实现的所有层都会这样去写.
class Embeddings(torch.nn.Module):
    def __init__(self, d_model, vocab):
        # 类的初始化函数, 有两个参数, d_model: 指词嵌入的维度, vocab: 指词表的大小.
        # 接着就是使用super的方式指明继承nn.Module的初始化函数, 我们自己实现的所有层都会这样去写.
        super(Embeddings, self).__init__()
        # 之后就是调用nn中的预定义层Embedding, 获得一个词嵌入对象self.lut
        self.lut = torch.nn.Embedding(vocab, d_model)
        # 最后就是将d_model传入类中
        self.d_model = d_model

    def forward(self, x):
        # 可以将其理解为该层的前向传播逻辑，所有层中都会有此函数
        # 当传给该类的实例化对象参数时, 自动调用该类函数
        # 参数x: 因为Embedding层是首层, 所以代表输入给模型的文本通过词汇映射后的张量

        # 将x传给self.lut并与根号下self.d_model相乘作为结果返回
        return self.lut(x) * math.sqrt(self.d_model)
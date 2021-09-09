import torch
from lib.MultiHeadedAttention import clones
from lib.LayerNorm import LayerNorm

# 使用Encoder类来实现编码器
class Encoder(torch.nn.Module):
    def __init__(self, layer, N):
        """初始化函数的两个参数分别代表编码器层和编码器层的个数"""
        super(Encoder, self).__init__()
        # 首先使用clones函数克隆N个编码器层放在self.layers中
        self.layers = clones(layer, N)
        # 再初始化一个规范化层, 它将用在编码器的最后面.
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """forward函数的输入和编码器层相同, x代表上一层的输出, mask代表掩码张量"""
        # 首先就是对我们克隆的编码器层进行循环，每次都会得到一个新的x，
        # 这个循环的过程，就相当于输出的x经过了N个编码器层的处理. 
        # 最后再通过规范化层的对象self.norm进行处理，最后返回结果. 
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
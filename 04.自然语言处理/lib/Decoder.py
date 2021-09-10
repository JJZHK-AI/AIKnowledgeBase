import torch
from lib.MultiHeadedAttention import clones
from lib.LayerNorm import LayerNorm

# 使用类Decoder来实现解码器
class Decoder(torch.nn.Module):
    def __init__(self, layer, N):
        """初始化函数的参数有两个，第一个就是解码器层layer，第二个是解码器层的个数N."""
        super(Decoder, self).__init__()
        # 首先使用clones方法克隆了N个layer，然后实例化了一个规范化层. 
        # 因为数据走过了所有的解码器层后最后要做规范化处理. 
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        """forward函数中的参数有4个，x代表目标数据的嵌入表示，memory是编码器层的输出，
           source_mask, target_mask代表源数据和目标数据的掩码张量"""

        # 然后就是对每个层进行循环，当然这个循环就是变量x通过每一个层的处理，
        # 得出最后的结果，再进行一次规范化返回即可. 
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)
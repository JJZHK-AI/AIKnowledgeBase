import torch
from lib.MultiHeadedAttention import clones
from lib.SublayerConnection import SublayerConnection

# 使用DecoderLayer的类实现解码器层
class DecoderLayer(torch.nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """初始化函数的参数有5个, 分别是size，代表词嵌入的维度大小, 同时也代表解码器层的尺寸，
            第二个是self_attn，多头自注意力对象，也就是说这个注意力机制需要Q=K=V， 
            第三个是src_attn，多头注意力对象，这里Q!=K=V， 第四个是前馈全连接层对象，最后就是droupout置0比率.
        """
        super(DecoderLayer, self).__init__()
        # 在初始化函数中， 主要就是将这些输入传到类中
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 按照结构图使用clones函数克隆三个子层连接对象.
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """forward函数中的参数有4个，分别是来自上一层的输入x，
           来自编码器层的语义存储变量mermory， 以及源数据掩码张量和目标数据掩码张量.
        """
        # 将memory表示成m方便之后使用
        m = memory

        # 将x传入第一个子层结构，第一个子层结构的输入分别是x和self-attn函数，因为是自注意力机制，所以Q,K,V都是x，
        # 最后一个参数是目标数据掩码张量，这时要对目标数据进行遮掩，因为此时模型可能还没有生成任何目标数据，
        # 比如在解码器准备生成第一个字符或词汇时，我们其实已经传入了第一个字符以便计算损失，
        # 但是我们不希望在生成第一个字符时模型能利用这个信息，因此我们会将其遮掩，同样生成第二个字符或词汇时，
        # 模型只能使用第一个字符或词汇信息，第二个字符以及之后的信息都不允许被模型使用.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # 接着进入第二个子层，这个子层中常规的注意力机制，q是输入x; k，v是编码层输出memory， 
        # 同样也传入source_mask，但是进行源数据遮掩的原因并非是抑制信息泄漏，而是遮蔽掉对结果没有意义的字符而产生的注意力值，
        # 以此提升模型效果和训练速度. 这样就完成了第二个子层的处理.
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        # 最后一个子层就是前馈全连接子层，经过它的处理后就可以返回结果.这就是我们的解码器层结构.
        return self.sublayer[2](x, self.feed_forward)
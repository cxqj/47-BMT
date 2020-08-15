import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, d_model, voc_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, voc_size)   # 300-->Vocab_Size
        print('Using vanilla Generator')

    def forward(self, x):
        '''
        Inputs:
            x: (B, Sc, Dc)
        Outputs:
            (B, seq_len, voc_size)
        '''
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)  # 注意最后需要加入softmax获取分类概率

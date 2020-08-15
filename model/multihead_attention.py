import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(Q, K, V, mask, dropout=None):
    # Q, K, V are (B, *(H), seq_len, d_model//H = d_k)
    # mask is     (B,    1,       1,               Ss)
    d_k = Q.size(-1)  # audio: (B,4,T_A,256) 
    # (B, H, S, S)
    QKt = Q.matmul(K.transpose(-1, -2))   #  audio: (B,4,T_A,256) x (B,4,256,T_A)-->(B,4,T_A,T_A) 
    sm_input = QKt / np.sqrt(d_k)   #  audio: (B,4,T_A,T_A) 

    # mask: (B,1,1,T_A)
    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))   # 将sm_input的最后一维数据置为负无穷

    softmax = F.softmax(sm_input, dim=-1)  #  audio: (B,4,T_A,T_A) 
    out = softmax.matmul(V)   # audio: (B,4,T_A,T_A) x (B,4,T_A,256)-->(B,4,T_A,256)

    if dropout is not None:
        out = dropout(out)

    # (B, *(H), seq_len, d_model//H = d_k)
    return out


class MultiheadedAttention(nn.Module):

    def __init__(self, d_model_Q, d_model_K, d_model_V, H, dout_p=0.0, d_model=None):
        super(MultiheadedAttention, self).__init__()
        self.d_model_Q = d_model_Q
        self.d_model_K = d_model_K
        self.d_model_V = d_model_V
        self.H = H
        self.d_model = d_model
        self.dout_p = dout_p

        if self.d_model is None:
            print(f'd_model: is None')
            self.d_model = self.d_model_Q

        self.d_k = self.d_model // H

        self.linear_Q2d = nn.Linear(self.d_model_Q, self.d_model)
        self.linear_K2d = nn.Linear(self.d_model_K, self.d_model)
        self.linear_V2d = nn.Linear(self.d_model_V, self.d_model)
        self.linear_d2Q = nn.Linear(self.d_model, self.d_model_Q)  # 将最后输出编码到原始维度

        self.dropout = nn.Dropout(self.dout_p)

        assert self.d_model % H == 0

    # 就拿音频特征举例，其余特征相同
    def forward(self, Q, K, V, mask):
        ''' 
            Q, K, V: (B, Sq, Dq), (B, Sk, Dk), (B, Sv, Dv)
            mask: (B, 1, Sk)
            Sk = Sv, 
            Dk != self.d_k
            Also: m1 is the target modality (queries); m2 is the source modality (keys, values)
        '''

        B, Sq, d_model_Q = Q.shape  #  audio: (B,T_A,128)   
        # (B, Sm, D) <- (B, Sm, Dm)
        Q = self.linear_Q2d(Q)    #  audio: (B,T_A,128)-->(B,T_A,1024)
        K = self.linear_K2d(K)    #  audio: (B,T_A,128)-->(B,T_A,1024)
        V = self.linear_V2d(V)    #  audio: (B,T_A,128)-->(B,T_A,1024)

        # (B, H, Sm, d_k) <- (B, Sm, D)  将特征划分为H个头
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2)  #audio: (B,T_A,1024)-->(B,T_A,4,256)-->(B,4,T_A,256) 
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)

        if mask is not None:
            # the same mask for all heads -> (B, 1, 1, Sm2)
            mask = mask.unsqueeze(1)   # audio: (B,1,T_A)-->(B,1,1,T_A)  

        # (B, H, Sq, d_k) <- (B, H, Sq, d_k), (B, H, Sk, d_k), (B, H, Sv, d_k), Sk = Sv
        Q = attention(Q, K, V, mask, self.dropout)   #audio: (B,4,T_A,256) 
        # (B, Sq, D) <- (B, H, Sq, d_k)
        Q = Q.transpose(-3, -2).contiguous().view(B, Sq, self.d_model)   #audio: (B,4,T_A,256)-->(B,T_A, 4, 256)-->(B,T_A,1024)
        # (B, Sq, Dq)
        Q = self.linear_d2Q(Q)  #audio: (B,T_A,1024)-->(B,T_A,128)  

        return Q  #audio:(B,T_A,128)  

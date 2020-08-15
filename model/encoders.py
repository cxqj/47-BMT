import torch
import torch.nn as nn

from model.blocks import (BridgeConnection, LayerStack,
                          PositionwiseFeedForward, ResidualConnection, clone)
from model.multihead_attention import MultiheadedAttention


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff):
        super(EncoderLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 2)
        self.self_att = MultiheadedAttention(d_model, d_model, d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dout_p=0.0)
        
    def forward(self, x, src_mask):
        '''
        in:
            x: (B, S, d_model), src_mask: (B, 1, S)
        out:
            (B, S, d_model)
        '''
        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs 
        # the output of the self attention
        sublayer0 = lambda x: self.self_att(x, x, x, src_mask)
        sublayer1 = self.feed_forward
        
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
        
        return x

# 双模态编码层
class BiModalEncoderLayer(nn.Module):
    # d_model_M1=128, d_model_M2=1024, d_model=1024, dout_p=0.1, H=4, d_ff_M1=512, d_ff_M2=4096
    def __init__(self, d_model_M1, d_model_M2, d_model, dout_p, H, d_ff_M1, d_ff_M2):
        super(BiModalEncoderLayer, self).__init__()  # 最终的输出维度和Q一样
        self.self_att_M1 = MultiheadedAttention(d_model_M1, d_model_M1, d_model_M1, H, dout_p, d_model)  # 音频特征自注意力机制   输出维度128
        self.self_att_M2 = MultiheadedAttention(d_model_M2, d_model_M2, d_model_M2, H, dout_p, d_model)  # 视频特征自注意力机制   输出维度1024
        self.bi_modal_att_M1 = MultiheadedAttention(d_model_M1, d_model_M2, d_model_M2, H, dout_p, d_model)  # 音频/视频注意力机制  输出维度128
        self.bi_modal_att_M2 = MultiheadedAttention(d_model_M2, d_model_M1, d_model_M1, H, dout_p, d_model)  # 视频/音频注意力机制  输出维度1024
        self.feed_forward_M1 = PositionwiseFeedForward(d_model_M1, d_ff_M1, dout_p)  # feed_forward层  128-->512-->128
        self.feed_forward_M2 = PositionwiseFeedForward(d_model_M2, d_ff_M2, dout_p)  # feed_forward层  1024-->4096-->1024
        self.res_layers_M1 = clone(ResidualConnection(d_model_M1, dout_p), 3)   # 残差连接  共三个残差连接  (自注意力，互注意力，feed_forward)
        self.res_layers_M2 = clone(ResidualConnection(d_model_M2, dout_p), 3)

    def forward(self, x, masks):
        '''
        Inputs:
            x (M1, M2): (B, Sm, Dm)
            masks (M1, M2): (B, 1, Sm)
        Output:
            M1m2 (B, Sm1, Dm1), M2m1 (B, Sm2, Dm2),
        '''
        # M1是音频特征，M2是视频特征
        M1, M2 = x   # M1: (B,T_A,128) M2: (B,T_V,1024)
        M1_mask, M2_mask = masks  # M1_mask： (B,1,T_A) M2_mask: (B,1,T_V)

        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs
        # the output of the self attention
        def sublayer_self_att_M1(M1): return self.self_att_M1(M1, M1, M1, M1_mask)
        def sublayer_self_att_M2(M2): return self.self_att_M2(M2, M2, M2, M2_mask)
        def sublayer_att_M1(M1): return self.bi_modal_att_M1(M1, M2, M2, M2_mask)
        def sublayer_att_M2(M2): return self.bi_modal_att_M2(M2, M1, M1, M1_mask)
        sublayer_ff_M1 = self.feed_forward_M1
        sublayer_ff_M2 = self.feed_forward_M2

        # 1. Self-Attention
        # both (B, Sm*, Dm*)
        # 调用的是上面的函数
        M1 = self.res_layers_M1[0](M1, sublayer_self_att_M1)  # 音频特征自注意力  输出：(B,T_A,128)
        M2 = self.res_layers_M2[0](M2, sublayer_self_att_M2)  # 视频特征自注意力  输出：(B,T_V,1024)

        # 2. Multimodal Attention (var names: M* is the target modality; m* is the source modality)
        # (B, Sm1, Dm1)
        M1m2 = self.res_layers_M1[1](M1, sublayer_att_M1)    # 音频/视频特征注意力  输出：(B,T_A,128)
        # (B, Sm2, Dm2)
        M2m1 = self.res_layers_M2[1](M2, sublayer_att_M2)    # 视频/音频特征注意力  输出：(B,T_V,1024)

        # 3. Feed-forward (var names: M* is the target modality; m* is the source modality)
        # (B, Sm1, Dm1)
        M1m2 = self.res_layers_M1[2](M1m2, sublayer_ff_M1)  # (B,T_A,128)-->(B,T_A,128*4)-->(B,T_A,128)
        # (B, Sm2, Dm2)
        M2m1 = self.res_layers_M2[2](M2m1, sublayer_ff_M2)  # (B,T_V,1024)-->(B,T_V,1024*4)-->(B,T_V,1024)

        return M1m2, M2m1  # (B,T_A,128), (B,T_V,1024)
    

class Encoder(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(Encoder, self).__init__()
        self.enc_layers = clone(EncoderLayer(d_model, dout_p, H, d_ff), N)
        
    def forward(self, x, src_mask):
        '''
        in:
            x: (B, S, d_model) src_mask: (B, 1, S)
        out:
            # x: (B, S, d_model) which will be used as Q and K in decoder
        '''
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x


class BiModalEncoder(nn.Module):
    # d_model_A: 128  d_model_V: 1024 d_model: 1024  dout_p: 0.1 H:4  d_ff_A: 4*128  d_ff_V: 4*1024 N:2
    def __init__(self, d_model_A, d_model_V, d_model, dout_p, H, d_ff_A, d_ff_V, N):
        super(BiModalEncoder, self).__init__()
        layer_AV = BiModalEncoderLayer(d_model_A, d_model_V, d_model, dout_p, H, d_ff_A, d_ff_V)
        self.encoder_AV = LayerStack(layer_AV, N)  # 两层编码层

    def forward(self, x, masks: dict):
        '''
        Input:
            x (A, V): (B, Sm, D)
            masks: {V_mask: (B, 1, Sv); A_mask: (B, 1, Sa)}
        Output:
            (Av, Va): (B, Sm1, Dm1)
        '''
        A, V = x   # A：(B,T_A,128) V: (B,T_V,1024)
 
        # M1m2 (B, Sm1, D), M2m1 (B, Sm2, D) <-
        Av, Va = self.encoder_AV((A, V), (masks['A_mask'], masks['V_mask']))  # Av:(B,T_A,128), Va:(B,T_V,1024)  

        return (Av, Va)  # Av:(B,T_A,128), Va:(B,T_V,1024)  

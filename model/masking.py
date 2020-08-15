import torch

def subsequent_mask(size):
    '''
    in: size
    out: (1, size, size)
    '''
    mask = torch.ones(1, size, size)
    mask = torch.tril(mask, 0)

    return mask.byte()

# 获取audio,video,caption特征不为1的mask
def mask(src, trg, pad_idx):  # SRC: (B,T)  TRG：(B,Seq_Len)  Pad_Idx: 1
    # masking the padding. src shape: (B, S') -> (B, 1, S')
    src_mask = (src != pad_idx).unsqueeze(1)  # (B,1,T)
    # 注意trg的mask是一个下三角矩阵，是为了保持循环的特征
    """
    True,False,False,....False
    True,True,False,.....False
    True,True,True,......False
    """
    if trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2) & subsequent_mask(trg.size(-1)).type_as(src_mask.data)
        return src_mask, trg_mask
    else:
        return src_mask

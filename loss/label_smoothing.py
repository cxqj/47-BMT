import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothing(nn.Module):
    
    def __init__(self, smoothing, pad_idx):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing  # 0.7
        self.pad_idx = pad_idx  # 1
        
    def forward(self, pred, target):  # pred (B, Seq_Len, Vocab_size), target (B, Seq_Len)
        # Note: preds are expected to be after log
        B, S, V = pred.shape
        # (B, S, V) -> (B * S, V); (B, S) -> (B * S)
        pred = pred.contiguous().view(-1, V)  # (B * Seq_Len, Vocab_size)
        target = target.contiguous().view(-1) # (B * Seq_Len)
        
        # prior (uniform)
        dist = self.smoothing * torch.ones_like(pred) / (V - 2)  # (B * Seq_Len, Vocab_size)
        # add smoothed ground-truth to prior (args: dim, index, src (value))
        # 第一个1表示沿着第一个维度，target.unsqueeze(-1)表示target对应的索引，1-self.smoothing表示填充的数字
        dist.scatter_(1, target.unsqueeze(-1).long(), 1-self.smoothing)  # 将dist中target对应的位置置为0.3    PyTorch 中，一般函数加下划线代表直接在原来的 Tensor 上修改
        # make the padding token to have zero probability 
        dist[:, self.pad_idx] = 0
        # ?? mask: 1 if target == pad_idx; 0 otherwise
        mask = torch.nonzero(target == self.pad_idx)  # target中pad_idx所在的索引，将pad_idx的dist置为0
        
        if mask.sum() > 0 and len(mask) > 0:
            # dim, index, val
            dist.index_fill_(0, mask.squeeze(), 0)   # 根据mask中的值将dist对应行置为0
            
        return F.kl_div(pred, dist, reduction='sum')

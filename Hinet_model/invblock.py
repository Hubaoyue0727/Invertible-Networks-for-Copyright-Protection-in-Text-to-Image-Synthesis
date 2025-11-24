from math import exp
import torch
import torch.nn as nn
import config as c
from Hinet_model.rrdb_denselayer import ResidualDenseBlock_out
from Hinet_model.Channel_atten import GAP_Linear,GAP_Conv


class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, subnet_channel_atten=GAP_Linear,clamp=c.clamp, harr=True, in_1=3, in_2=3):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
            # self.split_len1 = in_1 
            # self.split_len2 = in_2
        self.clamp = clamp
        # channel
        self.new = subnet_channel_atten(self.split_len2)
        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2
            w1 = self.new(y1)  
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * (w1 * x2) + t1  

        else:
            s1, t1 = self.r(x1), self.y(x1)
            w1 = self.new(x1)    # 修改
            y2 = (x2 - t1) / self.e(s1) / w1   
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)



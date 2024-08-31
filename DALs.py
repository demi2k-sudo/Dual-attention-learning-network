from torch import nn as nn
import torch
from DAL import DAL

class DALs(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, n_layers):
        super(DALs, self).__init__()
        self.dals = nn.ModuleList([DAL(hidden_size, num_heads, dropout) for _ in range(n_layers)])
        self.num_layers = n_layers

    def forward(self, x, y):
        for i in range(self.num_layers):
            x, y = self.dals[i](x, y)
        return torch.cat([x, y], dim=1)

# dalnet = DALs(768,12,0.2,5)

# x = torch.randn((1, 69, 768))
# y = torch.randn((1, 5, 768))
# print(dalnet(x,y).shape)
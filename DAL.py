import torch.nn as nn
from torch.nn import MultiheadAttention

class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[3].weight)
        nn.init.normal_(self.net[0].bias, std=1e-6)
        nn.init.normal_(self.net[3].bias, std=1e-6)

    def forward(self, x):
        return self.net(x)

class DAL(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(DAL, self).__init__()
        self.attention_left = MultiheadAttention(num_heads=num_heads, embed_dim=hidden_size, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)

        self.co_attention_left = MultiheadAttention(num_heads=num_heads, embed_dim=hidden_size, dropout=dropout)
        self.FFN_left = FeedForward(hidden_size, dropout)

        
        self.attention_right = MultiheadAttention(num_heads=num_heads, embed_dim=hidden_size, dropout=dropout)
        self.co_attention_right = MultiheadAttention(num_heads=num_heads, embed_dim=hidden_size, dropout=dropout)
        self.FFN_right = FeedForward(hidden_size, dropout)

    def forward(self, x, y):
        
        # print(self.attention_left(x, x, x)[1].shape)
        # print(y.shape)
        x = x.transpose(0, 1)  # (69, 1, 768)
        y = y.transpose(0, 1)  # (5, 1, 768)
        
        x = self.norm(x + self.attention_left(x, x, x)[0])
        y = self.norm(y + self.attention_right(y, y, y)[0])

        
        x = self.norm(x + self.co_attention_left(x, y, y)[0])
        y = self.norm(y + self.co_attention_right(y, x, x)[0])

        x = self.norm(x + self.FFN_left(x))
        y = self.norm(y + self.FFN_right(y))
        return x.transpose(0,1), y.transpose(0,1)
    
# dalnet = DAL(768,12,0.2)

# x = torch.randn((1, 69, 768))
# y = torch.randn((1, 5, 768))
# print(dalnet(x,y)[1].shape,dalnet(x,y)[0].shape)
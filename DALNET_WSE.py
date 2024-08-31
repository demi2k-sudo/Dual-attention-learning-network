import torch.nn as nn
from torch.nn import MultiheadAttention
from TSE import TSE
from ImageEncoder.model import ResnetCustom
from DALs import DALs


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
    

class DALNet_WSE(nn.Module):
    def __init__(self, dim, word_len, num_heads, dropout, n_layers, vocab_size, device):
        super().__init__()
        self.question_encoder = TSE(dim,word_len)

        self.img_encoder = ResnetCustom(dim)

        self.DALs = DALs(dim, num_heads, dropout, n_layers)

        self.fusion = nn.Linear(word_len+5, 1).to(device)
        self.activ1 = nn.Tanh()
        self.classifer = nn.Sequential(nn.Linear(dim, dim),
                                       nn.LayerNorm(dim, eps=1e-12, elementwise_affine=True),
                                       nn.GELU(),
                                       nn.Linear(dim, vocab_size),
                                       nn.Softmax(-1)).to(device)

    def forward(self, img, sens):
        # print("hello")
        question_embedding = self.question_encoder(sens)
        # print("question embedding done")
        visual_embedding = self.img_encoder(img)
        # print("image embedding done")
        z = self.DALs(visual_embedding, question_embedding)
        # print("DALs done")
        # print(z.shape)
        h = z.permute(0, 2, 1)  # torch.Size([16, 312, 25])
        # print(h.shape)
        pooled_h = self.activ1(self.fusion(h)).squeeze(2)
        # print(pooled_h.shape)
        logits = self.classifer(pooled_h)
        # print(logits.shape)
        return logits
    

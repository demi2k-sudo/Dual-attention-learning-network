from Embeddings import Embedder
from torch import nn
import torch


class TSE(nn.Module):
    def __init__(self, dim, word_len):
        super().__init__()
        self.WQ = nn.Linear(dim, dim)
        self.WK = nn.Linear(dim, dim)
        self.UQ = nn.Linear(dim, dim)
        self.UK = nn.Linear(dim, dim)
        self.dim = dim
        self.word_len = word_len
        self.embed = Embedder(word_len)
        self.V = nn.Linear(dim, dim)
        
    def forward(self, str):
        ES = self.embed.getSentenceEmbedding(str)
        print('Sentence Embedding : ',ES.shape)
        EW = self.embed.getWordEmbedding(str)
        print(EW.shape)
        
        # Compute query and key for words
        Q = self.WQ(EW)  # (batch_size, word_len, dim)
        K = self.WK(EW)  # (batch_size, word_len, dim)
        print(Q.shape,K.shape)
        
        # Compute query and key for sentences
        UQ = self.UQ(ES).unsqueeze(1)  # (batch_size, 1, dim)
        UK = self.UK(ES).unsqueeze(1)  # (batch_size, 1, dim)
        print(UQ.shape,UK.shape)
        
        Q = Q / (2 * self.dim**0.5)  
        K = K / (2 * self.dim**0.5)  
        UQ = (UQ / (2 * self.dim**0.5)).squeeze(1)  
        UK = (UK / (2 * self.dim**0.5)).squeeze(1)  
        print(UQ.shape,UK.shape)
        
        alpha = torch.bmm(Q, K.transpose(1, 2))  
        alpha += torch.bmm(UQ, UK.transpose(1, 2))
        print(alpha.shape)
        
        numerator = torch.exp(alpha)
        denominator = torch.sum(numerator, dim=-1).unsqueeze(1)
        print(numerator.shape, denominator.shape)
        value = self.V(EW)
        print(value.shape)
        value = value.view(-1,self.dim,self.word_len)
        soft = (numerator/denominator)
        result = value@soft
        return result.view(-1,self.word_len,self.dim)
        
        
# model = TSE(768,9)

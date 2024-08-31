
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1",cache_dir="F:\Research\Implementation\WordEmbedding\models")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1",cache_dir="F:\Research\Implementation\WordEmbedding\models")
word_embeddings = model.embeddings.word_embeddings

class WordEmbeddingModel(torch.nn.Module):
    
    def __init__(self, word_embeddings):
        super(WordEmbeddingModel, self).__init__()
        self.word_embeddings = word_embeddings

    def forward(self, input_ids):
        return self.word_embeddings(input_ids)

embedding_model = WordEmbeddingModel(word_embeddings)



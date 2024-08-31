from SentenceEmbedding.embed import model
from WordEmbedding.embed import embedding_model, tokenizer
import torch
import torch.nn.functional as F
device = "cuda"
# A class Embedder that has both sentence embedding model and word embedding models and with the necessary methods to trigger them


class Embedder:
    sentence_model = model
    word_model = embedding_model
    tokenizer = tokenizer

    def __init__(self, word_len):
        self.word_len = word_len

    # method to get Sentence Embeddings
    def getSentenceEmbedding(self, text):
        result = self.sentence_model.encode(text, convert_to_tensor=True)
        # print(type(result),result.shape)
        return result.unsqueeze(1)

    # method to get Word Embeddings
    def getWordEmbedding(self, texts):

        embeddings_list = []
        for text in texts:
            input_ids = self.tokenizer(text, return_tensors='pt')['input_ids']
            input_ids = F.pad(input_ids, (self.word_len -
                              len(input_ids[0]), 0), "constant", 0)

            with torch.no_grad():
                embeddings = self.word_model(input_ids)

            embeddings_list.append(embeddings.to(device))

        # Stack the list of tensors into a single tensor
        final_tensor = torch.stack(embeddings_list)

        return final_tensor.squeeze(1)

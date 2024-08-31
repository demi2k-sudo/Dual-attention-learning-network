import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "F:\Research\Implementation\SentenceEmbedding\models"


from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode("He drove to the stadium.")
# print(len(embeddings))
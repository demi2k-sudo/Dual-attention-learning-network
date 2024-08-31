from datasets import load_dataset
import os

os.environ["HF_DATASETS_CACHE"] = r"F:\Research\Implementation\Dataset\data"

ds = load_dataset("flaviagiammarino/vqa-rad")
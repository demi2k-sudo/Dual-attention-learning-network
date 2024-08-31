import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class MedicalVQADataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None, device='cpu'):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform
        self.device = device  # Added device attribute

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the image ID, question, and answer
        img_id = self.dataframe.iloc[idx]['image_id']
        question = self.dataframe.iloc[idx]['question']
        answer = self.dataframe.iloc[idx]['answer']

        # Load the image
        # Assuming the image extension is .jpg
        img_path = os.path.join(self.image_folder, f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')

        # Apply the transformation to convert image to tensor
        if self.transform:
            image = self.transform(image)

        # Move image to the specified device (e.g., cpu)
        image = image.to(self.device)

        return image, question, answer

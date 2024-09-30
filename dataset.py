import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from PIL import Image

class ChestXRay(Dataset):
    def __init__(self, df_dir, image_dir, transform=None):
        """
        Args:
            df_dir: Path to the csv file with image names and labels.
            image_dir: Directory with all the images with the labels.
            transform: Optional transform to be applied on a sample.
        """
        self.classes = ['Atelectasis',
                        'Cardiomegaly',
                        'Consolidation',
                        'Edema',
                        'Effusion',
                        'Emphysema',
                        'Fibrosis',
                        'Hernia',
                        'Infiltration',
                        'Mass',
                        'Nodule',
                        'Pleural_Thickening',
                        'Pneumonia',
                        'Pneumothorax']

        
        self.data_frame = pd.read_csv(df_dir)
        self.image_dir = image_dir
        self.transform = transform

        self.size = len(self.data_frame)
        self.labels = np.array(self.data_frame.iloc[:, 1:])
        self.images = self.data_frame.iloc[:,0]
        self.class_count = self.labels.sum(0)
        self.total_labels = self.class_count.sum()

    def __getitem__(self, idx):
        # Get image path
        img_name = os.path.join(self.image_dir, self.images.iloc[idx])
        image = Image.open(img_name).convert("RGB")
        
        # Get labels
        labels = np.array(self.labels[idx])
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        image = np.array(image)
        
        return {'image': image, 'labels': labels}
    
    def __len__(self):
        return self.size
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from PIL import Image

class ChestXRay(Dataset):
    def __init__(self, df_dir, image_dir, transform=None):
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
        img_name = os.path.join(self.image_dir, self.images.iloc[idx])
        image = Image.open(img_name).convert("RGB")
        
        labels = np.array(self.labels[idx])
        
        if self.transform:
            image = self.transform(image)

        image = np.array(image)
        
        return {'image': image, 'labels': labels}
    
    def __len__(self):
        return self.size
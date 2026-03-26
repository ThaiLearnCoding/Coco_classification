import torch
from torch.utils.data import Dataset, DataLoader, Subset
from zipfile import ZipFile
import json
from PIL import Image
from io import BytesIO
from torchvision import transforms
import numpy as np

class CocoMultimodalDataset(Dataset):
    def __init__(self, zip_path, target_classes, transform=None):
        self.zip_path = zip_path
        self.transform = transform
        self.target_classes = target_classes
        self.class_to_idx = {cls: i for i, cls in enumerate(target_classes)}
        
        with ZipFile(self.zip_path, 'r') as z:
            self.metadata = json.loads(z.read('metadata.json'))
        
        # Chỉ giữ lại các mẫu có nhãn nằm trong target_classes
        self.filtered_data = [
            m for m in self.metadata 
            if any(label in self.target_classes for label in m['labels'])
        ]

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        meta = self.filtered_data[idx]
        with ZipFile(self.zip_path, 'r') as z:
            img_data = z.read(f"images/{meta['file_name']}")
            img = Image.open(BytesIO(img_data)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # Lấy nhãn đầu tiên hợp lệ làm nhãn chính cho phân loại
        label_name = [l for l in meta['labels'] if l in self.target_classes][0]
        label_idx = self.class_to_idx[label_name]
        
        return img, label_idx

def get_dataloaders(config, clip_preprocess):
    full_dataset = CocoMultimodalDataset(
        zip_path=config['data']['zip_path'],
        target_classes=config['data']['classes'],
        transform=clip_preprocess
    )
    
    # Chia Train/Test
    n = len(full_dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(n * config['data']['train_split'])
    
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    train_loader = DataLoader(Subset(full_dataset, train_indices), 
                              batch_size=config['data']['batch_size'], shuffle=True)
    test_loader = DataLoader(Subset(full_dataset, test_indices), 
                             batch_size=config['data']['batch_size'], shuffle=False)
    
    return train_loader, test_loader

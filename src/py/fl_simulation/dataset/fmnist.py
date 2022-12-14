import torch, torchvision
from .dataset import Dataset
import os

class FMNISTDataset(Dataset): 
    
    def __init__(self, config):
        super(FMNISTDataset, self).__init__(config)
        self.labels = self.config.FMNIST_LABELS
        
    def load_train_data(self): 
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
        path = os.path.join(self.config.cwd, self.config.FMNIST_DATASET_PATH)
        train_dataset = torchvision.datasets.FashionMNIST(
            path, 
            train=True, download=True,
            transform=transform)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE_TRAIN, 
            shuffle=True)
        
        
        print("FashionMnist training data loaded.")
        return train_loader, train_dataset
    
    def load_test_data(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
        
        path = os.path.join(self.config.cwd, self.config.FMNIST_DATASET_PATH)
        test_dataset = torchvision.datasets.FashionMNIST(
            path, 
            train=False, download=True,
            transform=transform)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE_TEST, 
            shuffle=False)
        
        
        print("FashionMnist training data loaded.")
        return test_loader, test_dataset

        
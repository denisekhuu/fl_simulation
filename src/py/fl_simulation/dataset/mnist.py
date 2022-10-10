import torch, torchvision
from .dataset import Dataset
import os

class MNISTDataset(Dataset): 
    
    def __init__(self, config):
        super(MNISTDataset, self).__init__(config)
        
    def load_train_data(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.transforms.ToTensor()])
        path = os.path.join(self.config.cwd, self.config.MNIST_DATASET_PATH)
        train_dataset = torchvision.datasets.MNIST(
            path, 
            train=True, download=True,
            transform=transform)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE_TRAIN, 
            shuffle=True)
        
        print("MNIST training data loaded.")
        return train_loader, train_dataset
    
    def load_test_data(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.transforms.ToTensor()])
        path = os.path.join(self.config.cwd, self.config.MNIST_DATASET_PATH)
        test_dataset = torchvision.datasets.MNIST(
            self.config.MNIST_DATASET_PATH, 
            train=False, download=True,
            transform=transform)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE_TEST, 
            shuffle=False)
        
        print("MNIST test data loaded.")
        return test_loader, test_dataset
    

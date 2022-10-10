import os 
import torch.nn as nn
from torch import device
from ..nets import MNISTCNN, FMNISTCNN
from ..dataset import MNISTDataset, FMNISTDataset
from .configurations import Configurations


class DefaultConfigurations(Configurations): 
    """
    Default Configurations for a FL Simulation
    """
    
    def __init___():
        super(DefaultConfigurations).__init__()
        
    # Dataset Configurations
    BATCH_SIZE_TRAIN = 10
    BATCH_SIZE_TEST = 1000
    
    # MNIST_FASHION_DATASET Configurations
    FMNIST_NAME = "FMNIST"
    FMNIST_DATASET_PATH = os.path.join('temp/data/fmnist')
    FMNIST_LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',  'Bag', 'Ankle Boot']
    
    # MNIST_DATASET Configurations
    MNIST_NAME = "MNIST"
    MNIST_DATASET_PATH = os.path.join('temp/data/mnist')
    
    # Model Training Configurations
    ROUNDS = 200
    N_EPOCHS = 1
    LEARNING_RATE = 0.01
    MOMENTUM = 0.5
    LOG_INTERVAL = 10
    
    # Data Type Configurations
    DATASET = MNISTDataset
    MODELNAME = MNIST_NAME
    NETWORK = MNISTCNN
    NUMBER_TARGETS = 10
    
    # Temp Folder 
    TEMP = os.path.join('temp')
   
    # Local Environment Configurations
    NUMBER_OF_CLIENTS = 200
    DEVICE = device('cpu')
    
    # Default Server Configurations 
    CLIENTS_PER_ROUND = 5
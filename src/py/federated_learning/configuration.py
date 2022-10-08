import os
import torch.nn as nn
from torch import device
from .nets import MNISTCNN, FMNISTCNN
from .dataset import MNISTDataset, FMNISTDataset

class Configuration():
    
    # Dataset Config
    BATCH_SIZE_TRAIN = 10
    BATCH_SIZE_TEST = 1000
    
    #MNIST_FASHION_DATASET Configurations
    FMNIST_NAME = "FMNIST"
    FMNIST_DATASET_PATH = os.path.join('./data/fmnist')
    FMNIST_LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',  'Bag', 'Ankle Boot']
    
    #MNIST_DATASET Configurations
    MNIST_NAME = "MNIST"
    MNIST_DATASET_PATH = os.path.join('./data/mnist')
    
    #CIFAR_DATASET Configurations
    CIFAR10_NAME = "CIFAR10"
    CIFAR10_DATASET_PATH = os.path.join('./data/cifar10')
    CIFAR10_LABELS = ['Plane', 'Car', 'Bird', 'Cat','Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    #Model Training Configurations
    ROUNDS = 200
    N_EPOCHS = 1
    LEARNING_RATE = 0.01
    MOMENTUM = 0.5
    LOG_INTERVAL = 100
    
    # Data Type Configurations
    DATASET = MNISTDataset
    MODELNAME = MNIST_NAME
    NETWORK = MNISTCNN
    NUMBER_TARGETS = 10
    
    # Temp Folder 
    TEMP = os.path.join('./temp')
    
    #Local Environment Configurations
    NUMBER_OF_CLIENTS = 200
    CLIENTS_PER_ROUND = 5
    DEVICE = device('cpu')
    
    #Label Flipping Attack 
    POISONED_CLIENTS = 0
    DATA_POISONING_PERCENTAGE = 0
    FROM_LABEL = 0
    TO_LABEL = 0
    
    #Victoria Metrics Configurations
    VM_URL = os.getenv('VM_URL') #URL settings in docker-compose.yml
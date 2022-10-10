from abc import ABC, abstractmethod
from copy import deepcopy
from torch import load
from os.path import exists, join

class Client(ABC):
    
    def __init__(self, config, dataloader):
        """
        Parameters 
        :param config: experimental Configurations
        :type Configurations 
        :param dataloader: the torch dataloader for the client
        :type torchvision.datasets.Dataset
        :return:
        """
        self.config = config
        self.dataloader = dataloader
        self.net = self.load_default_model()
        
    def load_default_model(self, path = None):
        """
        Load a model from a file.
        The default behavior is to get the model in
        
        path: {working_dir}/{config.TEMP}/models/, "{config.MODELNAME}.model
        
        Parameters
        ---- 
        :param path (optional) : path to the default model
        :type path: string 
        
        :return: nn.Module
        ----
        
        """
        path = path if path else join(self.config.cwd, self.config.TEMP, 'models', "{}.model".format(self.config.MODELNAME))
        if not exists(path):
            self.config.root.error("Model not found. Check {}".format(path))
        else:
            try:
                model = self.config.NETWORK()
                model.load_state_dict(load(path))
                model.eval()
            except Exception as e:
                self.config.root.error("Error occured:  {}".format(e))
                self.config.root.error("Couldn't load model in {}.".format(path)) 
        return model
    
    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.
        :param new_params: New weights for the neural network
        :type new_params: dict
        :return:
        """
        self.net.load_state_dict(deepcopy(new_params), strict=True)
        self.net.eval()
        
    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        :return: collections.OrderedDict
        """
        return self.net.state_dict()
    
    @abstractmethod
    def train(self, epoch):
        """
        Train client model 
        Parameter 
        ----
        :param epoch: The current epoch
        :type epoch: int
        :return:
        ----
        """
        pass
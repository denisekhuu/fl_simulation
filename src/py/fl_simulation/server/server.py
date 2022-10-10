from os.path import join, exists
from torch import load
from abc import ABC, abstractmethod

class Server(ABC):
    
    def __init__(self, config):
        
        """
        Parameters 
        ---- 
        :param config: experiment configurations
        :type config: Configuration  
        :return: 
        ----
        
        """
        
        self.config = config
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
        
        Configuration Attributes
        ----
        :const MODELNAME: Name of the model which defines the name of the file
        :type MODELNAME: string  
        
        :const NETWORK: NN model
        :type NETWORK: nn.Module

        :const TEMP: Name of temporary directory with all temporary files 
        :type TEMP: string
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
                self.config.root.info("Load default model to server successfully.")
            except Exception as e:
                self.config.root.error("Error occured:  {}".format(e))
                self.config.root.error("Couldn't load model in {}.".format(path)) 
        return model
    
    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        Parameters
        ----
        :return: collections.OrderedDict
        ----
        """
        return self.net.state_dict()
    
    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.
        Parameters
        ----
        :param new_params: New weights for the neural network
        :type new_params: collections.OrderedDict
        
        :return:
        ----
        """
        self.net.load_state_dict(new_params, strict=True)
        
    @abstractmethod
    def aggregate_model(self, client_parameters, current_round): 
        """
        Aggregate new global model with client parameters 
        :param client_parameters: list of cleint model parameters 
        :type client_parameters: collections.OrderedDict[]
        """
        pass
        
    @abstractmethod
    def select_clients(self):
        """
        Select clients and return a list of client IDs
        Parameters
        ---- 
        :return: int[]
        ----
        """
        pass
        
        
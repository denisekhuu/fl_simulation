from abc import ABC, abstractmethod
class ClientPlane(ABC): 
    def __init__(self, config, data):
        """
        Simulation of isolated distributed clients. The local data should not leave the client plane.
        Parameters 
        ----
        :param config: experiment configurations
        :type config: Configuration
        :param data: aggregated dataset 
        :type data: torch.utils.dataset.Dataset
        :return:
        ----
        """
        self.config = config
        self.train_dataset = data.train_dataset
        self.train_dataloader = data.train_dataloader
        self.clients = []
        self.round = 0
        
    @abstractmethod    
    def create_clients(self):
        """
        Parameters 
        ----
        Create clients from dataloaders
        :return: Client[]
        ----
        """
        pass
    
    
    @abstractmethod    
    def update_selected_clients(self, selected_ids, new_parameters):
        """
        Update clients with new parameters 
        Parameters 
        ----
        :param selected_ids: list with IDs of clients 
        :type selected_ids: int[]
        :param new_parameters: new model parameters 
        :type new_parameters: collections.OrderedDict
        :return: 
        ----
        """
        pass
    
    
    @abstractmethod    
    def train_selected_clients(self, selected_ids):
        """
        Train selected clients and return new local parameters 
        :param selected_ids: list with IDs of clients 
        :type selected_ids: int[]
        :return: collections.OrderedDict[] 
        """
        pass
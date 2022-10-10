from torch.utils.data import Subset
from torch.utils.data import DataLoader
from .client_plane import ClientPlane
from ..client import DefaultClient

class DefaultClientPlane(ClientPlane):
    def __init__(self, config, data):
        super(DefaultClientPlane, self).__init__(config, data)
        self.clients = self.create_clients()
    
    def divide_data_equally(self):
        """
        Divides the dataset into NUMBER_OF_CLIENTS different subsets
        :return: torch.utils.data.Subset[]
        """
        indices = [[] for i in range(self.config.NUMBER_OF_CLIENTS)]
        for i in range(len(self.train_dataset)):
            indices[i % self.config.NUMBER_OF_CLIENTS].append(i)
        trainsets = [Subset(self.train_dataset, idx) for idx in indices]
        return trainsets

    def create_distributed_dataloaders(self, distributed_datasets):
        """
        Divides the dataset into NUMBER_OF_CLIENTS different subsets
        :param distributed_datasets : a list of torch subsets 
        :type torch.utils.data.Subset[] 
        
        :return: torch.utils.data.DataLoader[]
        """
        dataloaders = [
            DataLoader(client_set, batch_size=self.config.BATCH_SIZE_TRAIN,shuffle=True, num_workers=2)
            for client_set in distributed_datasets
        ]
        return dataloaders

    def create_clients(self):
        """
        Create clients from dataloaders
        return Client[]
        """
        distributed_datasets = self.divide_data_equally()
        distributed_dataloaders = self.create_distributed_dataloaders(distributed_datasets)
        self.config.root.info("Create {} clients with dataset of size {}".format(self.config.NUMBER_OF_CLIENTS, len(distributed_dataloaders[0].dataset)))
        return [DefaultClient(self.config, dataloader) for idx, dataloader in enumerate(distributed_dataloaders)]
            
    def update_selected_clients(self, selected_ids, new_parameters):
        """
        Update clients with new parameters 
        :param selected_ids: list with IDs of clients 
        :type int[]
        :param new_parameters: new model parameters 
        :type collections.OrderedDict
        """
        for client_id in selected_ids: 
            self.clients[client_id].update_nn_parameters(new_parameters)
        
    def train_selected_clients(self, selected_ids):
        """
        Train selected clients and return new local parameters 
        :param selected_ids: list with IDs of clients 
        :type int[]
        
        :return:  collections.OrderedDict[] 
        """
        for client_id in selected_ids: 
            for epoch in range(self.config.N_EPOCHS):
                self.clients[client_id].train(epoch)
                
        return [self.clients[client_id].get_nn_parameters() for client_id in selected_ids]
            
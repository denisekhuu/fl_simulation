from .server import Server
from .model_aggregator import ModelAggregator
from .client_selector import ClientSelector

class DefaultServer(Server):
    def __init__(self, config):
        super(DefaultServer, self).__init__(config)
        self.aggregator = ModelAggregator()
        self.selector = ClientSelector()

    def select_clients(self):
        return self.selector.random_selector(self.config.NUMBER_OF_CLIENTS, self.config.CLIENTS_PER_ROUND)

    def aggregate_model(self, client_parameters, current_round): 
        new_parameters = self.aggregator.federated_averaging(client_parameters)
        self.update_nn_parameters(new_parameters)
        self.config.root.info("Model aggregation in round {} was successful".format(current_round+1))
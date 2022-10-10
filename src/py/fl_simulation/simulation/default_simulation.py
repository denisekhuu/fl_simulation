from ..client_plane import DefaultClientPlane
from ..server import DefaultServer

class DefaultSimulation(): 
    def __init__(self, config, data):
        """
        Federated Learning Simulation using a client plane and a server.
        Parameters 
        ----
        :param config: experiment configurations
        :type config: Configuration
        :param data: aggregated dataset 
        :type data: torch.utils.dataset.Dataset
        :return: 
        ----
        """
        self.server = DefaultServer(config)
        self.client_plane = DefaultClientPlane(config, data)

    def run_round(self, current_round):
        """
        Run a single round of Federated Learning
        Parameters
        ----
        :param current_round: number of the current Federated Learning round
        :type current_round: int
        :return: 
        ----
        """
        selected_clients = self.server.select_clients()
        self.client_plane.update_selected_clients(selected_clients, self.server.get_nn_parameters())
        client_parameters = self.client_plane.train_selected_clients(selected_clients)
        self.server.aggregate_model(client_parameters, current_round)

    def run_simulation(self, n_rounds):
        """
        Run a Federated Learning simulation with n number of rounds
        Parameters
        ----
        :param n_rounds: number of learning rounds
        :type n_rounds: int
        :return: 
        ----
        """
        for i in range(n_rounds):
            self.run_round(i)
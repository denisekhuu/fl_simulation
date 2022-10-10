from torch.optim import SGD
from torch.nn.functional import nll_loss
from .client import Client

class DefaultClient(Client):
    
    def __init__(self, config, dataloader):
        """
        Parameters 
        ----
        :param config: experimental Configurations
        :type Configurations 
        :param dataloader: the torch dataloader for the client
        :type torchvision.datasets.Dataset
        :return:
        ----
        """
        super(DefaultClient, self).__init__(config, dataloader)
        self.optimizer = SGD(self.net.parameters(), lr=self.config.LEARNING_RATE, momentum=self.config.MOMENTUM)
        self.criterion = nll_loss
            
    def train(self, epoch):
        """
        Train client model 
        Parameter 
        ----
        :param current_round: The current number of rounds
        :type current_round: int
        :return:
        ----
        """
        self.net.train()
        for batch_idx, (data, target) in enumerate(self.dataloader):
            data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
            self.optimizer.zero_grad()
            output = self.net(data)
            out = output.log()
            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()
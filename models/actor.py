import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import hidden_init

class Actor(nn.Module):
    """ Actor (policy) model """

    def __init__(self, state_size, action_size, seed, layers):
        """ Initialize parameters and build the model
        
        Params
        =====
            state_size (int): size of the environment state
            action_size (int): size of the environment action
            seed (int): seed for the random
            layers (array[int]): array containing the size of each layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc_layers = [nn.Linear(state_size, layers[0])]

        for i in range(1, len(layers)):
            self.fc_layers.append(nn.Linear(layers[i-1], layers[i]))

        self.fc_layers.append(nn.Linear(layers[-1], action_size))

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.fc_layers) - 1):
            self.fc_layers[i].weight.data.uniform_(*hidden_init(self.fc_layers[i]))

        self.fc_layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """ Forward propagation on the Actor (policy) network, mapping states -> actions """
        for i in range(len(self.fc_layers) - 1):
            state = F.relu(self.fc_layers[i](state))
        
        return F.tanh(self.fc_layers[-1](state))

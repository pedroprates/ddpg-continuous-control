import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import hidden_init

class Critic(nn.Module):
    """ Critic (value) model """

    def __init__(self, state_size, action_size, seed, layers):
        """ Initialize parameters and build the model

        Params
        ======
            state_size (int): size of the environment state
            action_size (int): size of the environment action
            seed (int): seed for the random
            layers (array[int]): array containing the size of each layer
        """
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc_layers = [nn.Linear(state_size, layers[0]), nn.Linear(layers[0] + action_size, layers[1])]

        for i in range(2, len(layers)):
            self.fc_layers.append(nn.Linear(layers[i-1], layers[i]))

        self.fc_layers.append(nn.Linear(layers[-1], 1))

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.fc_layers) - 1):
            self.fc_layers[i].weight.data.uniform_(*hidden_init(self.fc_layers[i]))

        self.fc_layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """ Forward propagation on the Critic (value) network, mapping (state, action) pairs -> Q-values """
        state = F.relu(self.fc_layers[0](state))
        sa = torch.cat((state, action), dim=1)

        for i in range(1, len(self.fc_layers) - 1):
            sa = F.relu(self.fc_layers[i](sa))

        return self.fc_layers[-1](sa)

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import hidden_init

class Critic(nn.Module):
    """ Critic (value) model """

    def __init__(self, state_size, action_size, seed, l1, l2):
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

        self.fc1 = nn.Linear(state_size, l1)
        self.bn1 = nn.BatchNorm1d(l1)
        self.fc2 = nn.Linear(l1 + action_size, l2)
        self.fc3 = nn.Linear(l2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """ Forward propagation on the Critic (value) network, mapping (state, action) pairs -> Q-values """
        state = F.relu(self.bn1(self.fc1(state)))
        sa = torch.cat((state, action), dim=1)
        sa = F.relu(self.fc2(sa))

        return self.fc3(sa)

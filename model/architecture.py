from torch import nn


class Net(nn.Module):

    def __init__(self, state_space, action_space):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_space, action_space, bias=True),
            nn.Sigmoid(),
            nn.Linear(action_space, action_space, bias=True)
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('sigmoid'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        return self.model(x)

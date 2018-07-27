import torch
import torch.nn as nn
from torch.nn import functional as F


class IntroVAE(torch.nn.Module):
    def __init__(self, params, util):
        super(IntroVAE, self).__init__()
        self.params = params
        self.util = util
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.params.latent_dimension)
        self.fc22 = nn.Linear(400, self.params.latent_dimension)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = self.util.to_variable(torch.randn(std.size()))
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
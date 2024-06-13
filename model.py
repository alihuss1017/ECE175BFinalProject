import torch
from torch import nn

# inspired from the model.py script available in the discussion folder

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3a = nn.Linear(hidden_dim, latent_dim)
        self.fc3b = nn.Linear(hidden_dim, latent_dim)

        self.fc4 = nn.Linear(latent_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.fc2(self.fc1(x)))
        mu, sigma = self.fc3a(h), self.fc3b(h)
        return mu, sigma

    def decode(self, x):
        h = self.relu(self.fc5(self.fc4(x)))
        return torch.sigmoid(self.fc6(h))
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma * epsilon
        x_hat = self.decode(z_new)
        return x_hat, mu, sigma
    

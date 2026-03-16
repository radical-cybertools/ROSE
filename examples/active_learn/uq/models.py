import torch
import torch.nn as nn
import torch.nn.functional as F


# First example of non-deterministic model
class MC_Dropout_CNN(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(dropout_p)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# Second example of non-deterministic model
class MC_Dropout_MLP(nn.Module):
    def __init__(self, dropout_p=0.5, input_size=28 * 28, hidden_sizes=None, num_classes=10):
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = self.fc3(x)
        return x


# Bayesian base model
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters (mean and log-variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        self.prior_std = prior_std

    def forward(self, input):
        # Sample weights using reparameterization
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)

        eps_w = torch.randn_like(weight_std)
        eps_b = torch.randn_like(bias_std)

        weight = self.weight_mu + eps_w * weight_std
        bias = self.bias_mu + eps_b * bias_std

        return F.linear(input, weight, bias)

    def kl_divergence(self):
        # KL divergence between learned weight distribution and standard normal prior
        kld_weight = -0.5 * torch.sum(
            1 + self.weight_logvar - self.weight_mu.pow(2) - self.weight_logvar.exp()
        )
        kld_bias = -0.5 * torch.sum(
            1 + self.bias_logvar - self.bias_mu.pow(2) - self.bias_logvar.exp()
        )
        return kld_weight + kld_bias


# Bayesian model
class BayesianNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super().__init__()
        self.bfc1 = BayesianLinear(input_size, hidden_size)
        self.bfc2 = BayesianLinear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.bfc1(x))
        x = self.bfc2(x)
        return x

    def kl_loss(self):
        return self.bfc1.kl_divergence() + self.bfc2.kl_divergence()


def elbo_loss(output, target, kl_div, kl_weight=1.0):
    ce = F.cross_entropy(output, target)
    return ce + kl_weight * kl_div

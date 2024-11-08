import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Beta, RelaxedBernoulli, Bernoulli
from torch.distributions.kl import kl_divergence


class NetworkStructureSampler(nn.Module):
    """
    Samples the network structure Z from Beta-Bernoulli process prior
    Network structure Z represents depth + dropout mask as:
    (a) depth: the number of layers with activated neurons
    (b) binary mask modulating the neuron activations
    """

    def __init__(self, args, device: torch.device):
        super(NetworkStructureSampler, self).__init__()
        self.args = args
        self.device = device

        # epsilon to select the number of layers with activated neurons
        self.ε = args.epsilon

        # maximum number of neurons/channels (M) in each layer
        self.max_width = args.max_width

        # Truncation level for variational approximation
        self.truncation_level = args.truncation_level

        # Temperature for Concrete Bernoulli
        self.τ = torch.tensor(args.temp)

        # Number of samples to estimate expectations
        self.num_samples = args.num_samples

        # Hyper-parameters for prior beta
        self.α = torch.tensor(args.a_prior).float().to(self.device)
        self.β = torch.tensor(args.b_prior).float().to(self.device)

        # Define a prior beta distribution
        self.prior_beta = Beta(self.α, self.β)

        # inverse softplus to avoid parameter underflow
        α = np.log(np.exp(args.a_prior) - 1)
        β = np.log(np.exp(args.b_prior) - 1)

        # Define variational parameters for posterior distribution
        self.a_k = nn.Parameter(torch.Tensor(self.truncation_level).zero_() + α)
        self.b_k = nn.Parameter(torch.Tensor(self.truncation_level).zero_() + β)

    def get_variational_params(self):
        a_k = F.softplus(self.a_k) + 0.01
        b_k = F.softplus(self.b_k) + 0.01
        return a_k, b_k

    def get_kl(self):
        """
        Computes the KL divergence between variational beta and prior beta
        """
        a_k, b_k = self.get_variational_params()
        variational_beta = Beta(a_k, b_k)
        kl_beta = kl_divergence(variational_beta, self.prior_beta)
        return kl_beta.sum()

    def get_threshold(self, Z: torch.Tensor):
        """
        Compute the threshold i.e. layers with activated neurons

        Parameters
        ----------
        Z : binary mask matrix from beta-Bernoulli process

        Returns
        -------
        threshold: number of layers with activated neurons
        """

        # First, count the number of neurons in each layer
        threshold_Z = (Z > self.ε).sum(1)
        # Second, compute the layers with activated neurons
        threshold_array = (threshold_Z > 0).sum(dim=1).cpu().numpy()
        # Third, consider maximum of thresholds from multiple samples
        threshold = max(threshold_array)
        return threshold

    def forward(self, num_samples: int = 5, get_pi: bool = False):

        # Define variational beta distribution
        a_k, b_k = self.get_variational_params()
        variational_beta = Beta(a_k, b_k)

        # sample from variational beta distribution
        ν = variational_beta.rsample((num_samples,)).view(num_samples, self.truncation_level)  # S x K

        # Convert ν to π i.e. activation level of layer
        # Product of ν is equivalent to cumulative sum of log of ν
        π = torch.cumsum(ν.log(), dim=1).exp()

        keep_prob = π.detach().mean(0)
        π = π.unsqueeze(1).expand(-1, self.max_width, -1)  # S x M x K

        # sample binary mask z_l given the activation level π_l of the layer
        if self.training:
            # draw continuous binary sample from concrete Bernoulli distribution to backpropagate the gradient
            concrete_bernoulli_dist = RelaxedBernoulli(probs=π, temperature=self.τ)
            Z = concrete_bernoulli_dist.rsample()
        else:
            # draw discrete sample from Bernoulli distribution
            bernoulli_dist = Bernoulli(probs=π)
            Z = bernoulli_dist.sample()

        threshold = self.get_threshold(Z)

        if get_pi:
            # return probabilities to plot
            return Z, threshold, keep_prob

        return Z, threshold

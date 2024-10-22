import torch
from torch import nn
from src.layers.layers import MLPBlock
from src.models.NetworkStructureSampler import NetworkStructureSampler


class AdaptiveMLP(nn.Module):
    def __init__(self, input_feature_dim, out_feature_dim, args, device):
        super(AdaptiveMLP, self).__init__()
        self.args = args
        self.input_feature_dim = input_feature_dim
        self.out_feature_dim = out_feature_dim

        # Maximum number of neurons (M)
        self.max_width = args.max_width

        # Truncation level for variational approximation
        self.truncation_level = args.truncation_level # K truncation_level
        self.num_samples = args.num_samples
        self.device = device

        # define neural network structure sampler with parameters defined in argument parser
        self.structure_sampler = NetworkStructureSampler(args, self.device)

        # Define the architecture of neural network by simply adding K layers at initialization
        # Note: we can also dynamically add new layers based on the inferred depth
        self.layers = nn.ModuleList([MLPBlock(self.input_feature_dim, self.max_width)])
        for i in range(self.truncation_level):
            self.layers.append(MLPBlock(self.max_width, self.max_width, residual=True))

        self.out_layer = nn.Linear(self.max_width, self.out_feature_dim)

    def _forward(self, x, mask_matrix, threshold):
        """
        Transform the input feature matrix with a sampled network structure
        Parameters
        ----------
        x : input feature matrix
        mask_matrix : mask matrix from Beta-Bernoulli Process
        threshold : number of layers in sampled structure

        Returns
        -------
        out : Output from the sampled architecture
        """

        # if number of layers sampled during testing is greater than truncation level, we use the network with
        # truncation level.
        if not self.training and threshold > len(self.layers):
            threshold = len(self.layers)

        for layer in range(threshold):
            mask = mask_matrix[:, layer]
            x = self.layers[layer](x, mask)

        out = self.out_layer(x)
        return out

    def forward(self, x, num_samples):
        """
        Transforms the input feature matrix with different samples of network structures Z

        Parameters
        ----------
        x : data
        num_samples : Number of samples of network structure Z

        Returns
        -------
        act_vec : Tensor by stacking the output from different structures Z
        """
        act_vec = []
        Z, threshold = self.structure_sampler(num_samples)

        for s in range(num_samples):
            out = self._forward(x, Z[s], threshold)
            act_vec.append(out.unsqueeze(0))

        act_vec = torch.cat(act_vec, dim=0)
        return act_vec

    def get_E_loglike(self, neg_loglike_fun, act_vec, y):
        """
        Compute the expectation of log likelihood
        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        act_vec : scores from different samples of latent neural architecture Z
        y : real labels

        Returns
        -------
        mean_neg_loglike: Expectation of negative log likelihood of the model based on different structures Z
        """
        y = y.expand(self.num_samples, -1, -1)
        neg_log_likelihood = neg_loglike_fun(act_vec[:, :, 0], y.squeeze())
        mean_neg_log_likelihood = neg_log_likelihood.mean(0).mean()
        return mean_neg_log_likelihood

    def estimate_ELBO(self, neg_loglike_fun, act_vec, y, N_train, kl_weight=1):
        """
        Estimate the ELBO
        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        act_vec : scores from different samples of latent neural architecture Z
        y : real labels
        N_train: number of training data points
        kl_weight: coefficient to scale KL component

        Returns
        -------
        ELBO
        """
        E_loglike = self.get_E_loglike(neg_loglike_fun, act_vec, y)
        KL = self.structure_sampler.get_kl()
        ELBO = E_loglike + (kl_weight * KL)/N_train
        return ELBO
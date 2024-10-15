import torch
from torch import nn
from src.layers.layers import MLPBlock, ConvBlock
from src.models.NetworkStructureSampler import NetworkStructureSampler


class AdaptiveConvNet(nn.Module):
    def __init__(self, input_channels, num_classes, num_channels=32, kernel_size=3, args=None, device=None):
        super(AdaptiveConvNet, self).__init__()
        self.mode = "NN"
        self.args = args

        self.max_channels = num_channels
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.truncation_level = args.truncation_level
        self.num_samples = args.num_samples
        self.device = device

        self.structure_sampler = NetworkStructureSampler(args, self.device)

        self.layers = nn.ModuleList([ConvBlock(self.input_channels, self.max_channels,
                                      kernel_size=5, pool=True).to(self.device)])

        for i in range(1, self.truncation_level):
            self.layers.append(ConvBlock(self.max_channels,
                                         self.max_channels,
                                         kernel_size=3,
                                         padding=1,
                                        residual=True).to(self.device))

        self.out_layer = nn.Sequential(MLPBlock(self.max_channels, self.max_channels, residual=True),
                                       nn.Linear(self.max_channels, self.num_classes))


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

        x = x.mean(dim=(2, 3))
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

    def get_E_loglike(self, neg_loglike_fun, output, target):
        """

        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        output : scores predicted by model
        target : real labels

        Returns
        -------
        mean negative log likelihood of the model based on different architectures
        """

        num_samples = self.num_samples

        batch_sze = target.shape[0]
        target_expand = target.repeat(num_samples)
        output = output.view(num_samples * batch_sze, -1)
        neg_loglike = neg_loglike_fun(output, target_expand).view(num_samples, batch_sze)
        E_neg_loglike = neg_loglike.mean(0).mean()
        return E_neg_loglike

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
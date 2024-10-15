import torch
from torch import nn as nn
from torch.distributions import Bernoulli


def conv3x3(in_channels: int, out_channels: int, kernel_size=3, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=False)


class global_mean_pool(nn.Module):
    """
    Takes average of image
    """

    def __init__(self):
        super(global_mean_pool, self).__init__()

    def forward(x):
        return x.mean(dim=(2, 3))


class MLPBlock(nn.Module):
    """
    Mask the output of MLP with binary vectors from Beta-Bernoulli prior and add residual
    """

    def __init__(self, in_neurons, out_neurons, residual=False):
        super(MLPBlock, self).__init__()

        self.linear = nn.Linear(in_neurons, out_neurons)
        self.act = nn.LeakyReLU()
        # self.bn = nn.BatchNorm1d(out_neurons, track_running_stats=False)
        self.bn = nn.GroupNorm(1, out_neurons)
        self.residual = residual

    def forward(self, x, mask=None):
        output = self.bn(self.act(self.linear(x)))

        if mask is not None:
            output *= mask.view(1, -1)

        if self.residual:
            return output + x
        else:
            return output


class ConvBlock(nn.Module):
    """
    Mask the output of CNN with binary vectors from Beta-Bernoulli prior and add residual
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=False, residual=False, drop=None):
        super(ConvBlock, self).__init__()

        self.conv_layer = conv3x3(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.act = nn.LeakyReLU()
        # self.bn = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.bn = nn.GroupNorm(1, out_channels)
        self.pool = pool
        self.drop = drop
        
        if self.drop != None:
            self.dropact = nn.Dropout2d(p=self.drop)
        
        if pool:
            self.pool_layer = nn.AvgPool2d(2, 2)

        self.residual = residual

        self.downsample = False
        if out_channels != in_channels and residual:
            self.downsample = True
            self.downsample_conv_layer = conv1x1(in_channels, out_channels, stride=2, padding=padding)
            self.downsample_norm_layer = nn.BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self, x, mask=None):
        print(x.shape)
        if self.drop == None:
            output = self.bn(self.act(self.conv_layer(x)))
        else:
            output = self.bn(self.dropact(self.act(self.conv_layer(x))))

        print(output.shape)
        if self.pool:
            output = self.pool_layer(output)
        
        print(mask.shape)
        if mask is not None:
            mask = mask.view(1, mask.shape[0], 1, 1)
            print(mask.shape)
            output *= mask
            print(output.shape)
        exit(0)

        if self.residual:
            if self.downsample:
                residual = self.downsample_norm_layer(self.downsample_conv_layer(x))
                output += residual
            else:
                output += x

        return output


class StochasticDepthConvBlock(nn.Module):
    """
    Mask the output of CNN with binary vectors from Beta-Bernoulli prior and add residual
    """

    def __init__(self, in_channels, out_channels, prob, multFlag, kernel_size=3, stride=1, padding=1):
        super(StochasticDepthConvBlock, self).__init__()

        self.conv_layer = conv3x3(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.prob = prob
        self.multFlag = multFlag
        self.bernoulli = Bernoulli(probs=torch.Tensor([self.prob]))

    def forward(self, x):
        identity = x.clone()

        if self.training:
            m = self.bernoulli.sample()
            if m == 1:
                self.conv_layer.weight.requires_grad = True

                out = self.conv_layer(x)
                out = self.act(out)
                out = self.bn(out)

                out += identity

            else:
                self.conv_layer.weight.requires_grad = False
                out = identity
        else:
            out = self.conv_layer(x)
            out = self.act(out)
            out = self.bn(out)

            if self.multFlag:
                out = self.prob * out + identity
            else:
                out = out + identity

        return out
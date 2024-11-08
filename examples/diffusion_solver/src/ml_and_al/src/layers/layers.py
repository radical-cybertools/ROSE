from torch import nn as nn


def conv3x3(in_channels: int, out_channels: int, kernel_size=3, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """
    2D Convolutional Layer with stride and padding
    Parameters
    ----------
    in_channels : Number of input channels
    out_channels : Number of output channels
    kernel_size : Size of kernel
    stride : Stride of the convolution
    padding : Padding added to all four sides of the input

    Returns
    -------
    Torch Conv2d module
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """
    2D Convolutional Layer with 1x1 kernel
    Parameters
    ----------
    in_channels : Number of input channels
    out_channels : Number of output channels
    stride : Stride of the convolution
    padding : Padding added to all four sides of the input

    Returns
    -------
    Torch Conv2d module
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=False)


class global_mean_pool(nn.Module):
    """
    Global averaging of the image
    """

    def __init__(self):
        super(global_mean_pool, self).__init__()

    def forward(self, x):
        """
        Computes global average of image
        Parameters
        ----------
        x : input feature map (Batch_size * Channels * Width * Height)

        Returns
        -------
        feature vector (Batch_size * Channels)
        """
        return x.mean(dim=(2, 3))


class MLPBlock(nn.Module):
    """
    Mask the output of fully connected layer with binary vectors from Beta-Bernoulli prior and add residual
    """

    def __init__(self, in_neurons, out_neurons, residual=False):
        super(MLPBlock, self).__init__()

        self.linear = nn.Linear(in_neurons, out_neurons)
        self.act_layer_fn = nn.LeakyReLU()
        self.norm_layer = nn.BatchNorm1d(out_neurons)
        self.residual = residual

    def forward(self, x, mask=None):
        """
        Transforms x, applies mask and add residual
        Parameters
        ----------
        x : input feature matrix (Batch_size * out_neurons)
        mask : binary vector to mask the feature matrix (out_neurons)

        Returns
        -------
        Output of the layer (Batch_size * out_neurons)
        """

        residual = x

        out = self.act_layer_fn(self.linear(x))
        out = self.norm_layer(out)

        if mask is not None:
            out *= mask.view(1, -1)

        if self.residual:
            out += residual

        return out


class ConvBlock(nn.Module):
    """
    Mask the output of CNN with binary vectors from Beta-Bernoulli prior and add residual
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=False, residual=False, act="leaky", dropout=False):
        super(ConvBlock, self).__init__()

        self.conv_layer = conv3x3(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        if act == "leaky":
            self.act_layer = nn.LeakyReLU(0.02)
        else:
            self.act_layer = nn.ReLU()
        self.norm_layer = nn.BatchNorm2d(out_channels)
        self.pool = pool
        self.drop = dropout
        if pool:
            self.pool_layer = nn.AvgPool2d(2, 2)
            
        if self.drop:
            self.drop_layer = nn.Dropout2d(0.4)

        self.residual = residual

    def forward(self, x, mask=None):
        """
        Transforms feature matrix x, applies mask and add residual
        Parameters
        ----------
        x : input features (Batch_size * out_channels * width * height)
        mask : binary vector to mask the feature matrix (out_channels)

        Returns
        -------
        If pool is false:
            Output of the layer (Batch_size * out_channels * width * height)
        else:
            Output of the layer (Batch_size * out_channels * width' * height')
        """
        residual = x

        output = self.act_layer(self.conv_layer(x))
        if self.drop:
            output = self.drop_layer(x)
            
        output = self.norm_layer(output)

        if self.pool:
            output = self.pool_layer(output)

        if mask is not None:
            mask = mask.view(1, mask.shape[0], 1, 1)
            output *= mask

        if self.residual:
            output += residual

        return output
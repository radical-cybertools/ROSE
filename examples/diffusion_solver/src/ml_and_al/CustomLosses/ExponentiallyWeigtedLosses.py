from torch import nn, abs, ones_like, exp, mean


class ExponentialMeanError(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean', difference_exponent=1, exponential_weight=1):
        super(ExponentialMeanError, self).__init__()

        self.sa = size_average

        self.reduce = reduce

        self.reduction = reduction

        self.alph = difference_exponent

        self.w = exponential_weight

    def mean_error(self, output, target):

        loss = mean(exp(-abs(ones_like(output) - output) / self.w) * abs((output - target) ** self.alph))

        return loss

    def forward(self, output, target):
        return self.mean_error(output, target)

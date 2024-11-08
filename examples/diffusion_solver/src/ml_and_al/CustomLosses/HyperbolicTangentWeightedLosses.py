from torch import nn, abs, tanh, mean


class HyperbolicTangentMeanError(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean', difference_exponent=1, target_weight=1,
                 hyperbolic_weight=2000):
        super(HyperbolicTangentMeanError, self).__init__()

        self.sa = size_average
        self.reduce = reduce
        self.reduction = reduction

        self.alph = difference_exponent
        self.w = target_weight
        self.w2 = hyperbolic_weight

    def mean_error(self, output, target):
        loss = mean((1 + tanh(self.w * target) * self.w2) * abs((output - target) ** self.alph))
        return loss

    def forward(self, output, target):
        return self.mean_error(output, target)

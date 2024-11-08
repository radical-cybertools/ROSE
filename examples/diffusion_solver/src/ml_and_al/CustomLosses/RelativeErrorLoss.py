from torch import nn, abs


class RelativeErrorLoss(nn.Module):
    r"""
    Implement relative error: https://en.wikipedia.org/wiki/Approximation_error
    Error = abs(target - prediction / target)
    To avoid division by 0 we add epsilon << 1 to the denominator.
    This error by default **does not** averages the error of all images, it returns the error for each image separately.

    Arguments: size_average, reduce, reduction are the same as pytorch's L1Loss, their explanation, from
    https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html - accessed in 2021-11-05 - is copied below. The new
    argument 'eps' follows.
    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'none'``
        eps (float, optional): Number to be added to the denominator of the relative error. Default: 1e-10

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(*)`, same shape as the input.

    """
    def __init__(self, size_average=None, reduce=None, reduction='none', eps=1e-10):
        super(RelativeErrorLoss, self).__init__()

        self.sa = size_average
        self.reduce = reduce
        self.reduction = reduction

        self.eps = eps

        self.mae = nn.L1Loss(size_average=self.sa, reduce=self.reduce, reduction=self.reduction)

    def forward(self, x, y):
        return self.mae(x, y) / (abs(x) + self.eps)

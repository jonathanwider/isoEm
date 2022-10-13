import torch.nn.functional as F
import torch

def plane_group_spatial_max_pooling(x, ksize, stride=None, pad=0):
    xs = x.size()
    x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
    x = F.max_pool2d(input=x, kernel_size=ksize, stride=stride, padding=pad)
    x = x.view(xs[0], xs[1], xs[2], x.size()[2], x.size()[3])
    return x

def plane_group_spatial_orientational_max_pooling(x, stride=None, pad=0):
    """
    Assume input x has shape (batchsize, n_channels*n_stabilizer, n_charts*height, width).
    Careful: This can be applied only if stabilizer_size is 6.
    Always maxpool over the whole image
    """
    xs = x.size()
    x = x.view(xs[0], -1, 6 * xs[2], xs[3]) # reshape such that orientation channels get pooled too.
    x = F.max_pool2d(input=x, kernel_size=(6*xs[2], xs[3]), stride=stride, padding=pad)

    x = x.view(xs[0], xs[1])
    return x

def plane_group_orientational_max_pooling(x,stride=None, pad=0):
    xs = x.size()
    assert(len(xs) == 5)
    x = x.permute((0,1,3,4,2)).contiguous()
    x = x.view(xs[0]*xs[1]*xs[3], xs[4], 6)
    x = F.max_pool1d(input=x, kernel_size=6, stride=stride, padding=pad)
    x = x.view(xs[0], xs[1], xs[3], xs[4])
    return x
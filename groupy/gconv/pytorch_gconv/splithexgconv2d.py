from groupy import hexa
import groupy.gconv.pytorch_gconv.splitgconv2d as splitgconv2d
import torch
from torch.nn import Parameter

class SplitHexGConv2D(splitgconv2d.SplitGConv2D):
    """
    """
    def __init__(self, *args, **kwargs):
        """
        """
        super(SplitHexGConv2D, self).__init__(*args, **kwargs)

        filter_mask = hexa.mask.hexagon_axial(self.ksize)
        filter_mask = filter_mask[None, None, None, ...]# .astype(self.dtype)
        # self.filter_mask = torch.from_numpy(filter_mask)
        self.register_parameter('filter_mask', Parameter(torch.from_numpy(filter_mask), requires_grad=False))

    def __call__(self, x):
        # Apply a mask to the parameters
        self.weight.data = self.weight.data * self.filter_mask

        y = super(SplitHexGConv2D, self).__call__(x)

        # Get a square shaped mask if it does not yet exist.
        """
        # we do not need this part because our feature maps already fill all image.
        if not hasattr(self, 'output_mask'):
            ny, nx = y.data.shape[-2:]
            self.output_mask = torch.from_numpy(hexa.mask.square_axial(ny, nx)[None, None, None, ...])

        y = y * self.output_mask
        """

        return y

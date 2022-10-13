# Define all the modules we want to use in our UNet here.

import torch
import torch.nn as nn
import torch.nn.functional as F

from groupy.gconv.pytorch_gconv.p6_conv_axial import P6ConvZ2, P6ConvP6
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling, \
     plane_group_orientational_max_pooling

from torch.nn import BatchNorm3d as IcoBatchNorm2d


def g_padding_full(t, in_stab_size):
    """
    Do G-Padding. Assume the input-data already has the shape "with padding". t should be of shape
    (batch_size, n_channels*(in_stab_size), n_charts*(height_chart+padding), (width_chart+padding))
    """

    bs = t.shape[0]
    assert in_stab_size in [1, 6]  # check that the input stabilizer size is valid
    h = int(t.shape[-2] / 5 - 2)  # true height of a single map without padding
    w = int(t.shape[-1] - 2)  # true width of a single map without padding
    n_charts = 5

    # split the tensor into the individual charts (at least in view)
    t_v = t.view(bs, -1, in_stab_size, 5, h + 2, w + 2)

    # Set invalid points to zero

    t_v[..., 0, 0:2] = 0  # set single pixels to zero.
    t_v[..., -1, 0] = 0  # set single pixel to zero.
    t_v[..., -1, -1] = 0  # set single pixel to zero.
    # set "corner-points" of icosahedron on the inside of the chart to zero
    t_v[..., -2, 1::h] = 0

    # Pad edges

    # left edge - shift orientation by +1 by rot of edge cw
    t_v[..., 1:-1, 0] = torch.roll(t_v[..., 1, 1:h + 1], shifts=(1, -1), dims=(-3, -2))
    # upper left edge - shift orientation by -1 by rot of edge ccw
    t_v[..., 0, 2:h + 2] = torch.roll(t_v[..., 1:-1, 1], shifts=(-1, 1), dims=(-3, -2))
    # upper right edge
    t_v[..., 0, h + 1:-1] = torch.roll(t_v[..., -2, 1:h + 1], shifts=(1,), dims=(-2,))
    # right edge - shift orientation by +1 bc rot of padded edge cw
    t_v[..., 0:-2, -1] = torch.roll(t_v[..., -2, h + 1:-1], shifts=(1, 1), dims=(-3, -2))
    # lower left edge
    t_v[..., -1, 1:h + 1] = torch.roll(t_v[..., 1, h + 1:-1], shifts=(-1,), dims=(-2,))
    # lower right edge - shift orientation by -1 bc rot of padded edge ccw
    t_v[..., -1, h + 1:-1] = torch.roll(t_v[..., 1:-1, -2], shifts=(-1, -1), dims=(-3, -2))


class IcoBatchNorm2d(nn.Module):
    def __init__(self, n_ch_in):
        """
        Instantiate BN and assign as member variable
        """
        super(IcoBatchNorm2d, self).__init__()
        self.n_ch_in = n_ch_in
        self.BN = nn.BatchNorm2d(n_ch_in)

    def forward(self, x):
        # tensor x has shape (batchsize, n_channels, n_stabilizer, n_charts*height, width)
        bs, n_in, n_stab, h, w = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(bs * n_stab, n_in, h, w)
        x = self.BN(x)
        x = x.view(bs, n_stab, n_in, h, w)
        x = x.transpose(1, 2).contiguous()
        return x


class PadP6ConvZ2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PadP6ConvZ2, self).__init__()
        self.conv = P6ConvZ2(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=3,
                             padding=1)

    def forward(self, x):
        # x = x.view(x.shape[0], x.shape[1], 1, 5, -1, x.shape[-1])
        # x = F.pad(x, (1, 1, 1, 1))
        # x = x.view(x.shape[0], x.shape[1], 1, -1, x.shape[-1])
        # convolution 1
        g_padding_full(x, in_stab_size=1)  # modifies x
        x = self.conv(x)
        return x


class InPadP6ConvZ2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InPadP6ConvZ2, self).__init__()
        self.conv = P6ConvZ2(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=3,
                             padding=1)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 5, -1, x.shape[-1])
        x = F.pad(x, (1, 1, 1, 1))
        x = x.view(x.shape[0], x.shape[1], 1, -1, x.shape[-1])
        # convolution 1
        g_padding_full(x, in_stab_size=1)  # modifies x
        x = self.conv(x)
        return x


class OutPadP6ConvZ2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutPadP6ConvZ2, self).__init__()
        self.conv = P6ConvP6(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=3,
                             padding=1)

    def forward(self, x):
        """
        Because we have g-padding the stride-convolution is not trivial.
        We need to add rows in order to maintain the right shape.
        We do this by adding one row at the bottom of each chart. Afterwards we also need to g_pad the results.
        Assume x has shape (batch_size, n_channels, n_stabilizer, n_charts*height, width)
        """
        g_padding_full(x, 6)
        x = self.conv(x)  # don't apply ReLU and BN here
        x = plane_group_orientational_max_pooling(x)  # pool over all orientations.
        x = x.view(x.shape[0], x.shape[1], 5, -1, x.shape[-1])
        x = x[..., 1:-1, 1:-1]
        x = x.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])
        return x


class PadP6ConvP6(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PadP6ConvP6, self).__init__()
        self.conv = P6ConvP6(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=3,
                             padding=1)

    def forward(self, x):
        """
        Because we have g-padding the strided convolution is not trivial.
        We need to add rows in order to maintain the right shape.
        We do this by adding one row at the bottom of each chart. Afterwards we also need to g_pad the results.
        Assume x has shape (batchsize, n_channels, n_stabilizer, n_charts*height, width)
        """
        g_padding_full(x, 6)
        x = self.conv(x)
        return x


class UpSampleIco(nn.Module):
    def __init__(self, in_res):
        super(UpSampleIco, self).__init__()
        self.w_pad_in = 2**(in_res)+2
        self.h_pad_in = 5*2**(in_res-1)+10
        self.up = nn.Upsample(size=(self.h_pad_in*2-1, self.w_pad_in*2-1), mode='bilinear', align_corners=True)

    def forward(self, x):
        """
        Because we have g-padding the strided convolution is not trivial.
        We need to add rows in order to maintain the right shape.
        We do this by adding one row at the bottom of each chart. Afterwards we also need to g_pad the results.
        Assume x has shape (batchsize, n_channels, n_stabilizer, n_charts*height, width)
        """
        bs, n_ch, n_stab, fh, w = x.shape
        x = x.view((bs, n_ch*n_stab, fh, w))  # combine channels and stabilizer dim
        x = self.up(x)
        x = F.pad(x, (0, 1, 1, 0))  # add single line on top and on the right for symmetry reasons
        x = x.view(bs, n_ch, n_stab, 5, -1, x.shape[-1])
        # remove excess from up-sampling the padding
        x = x[..., 1:-1, 1:-1]
        x = x.reshape(bs, n_ch, n_stab, 2*fh-2*5, 2*w-2)
        return x


class DownSampleIco(nn.Module):
    def __init__(self):
        super(DownSampleIco, self).__init__()
        self.down = torch.nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        """
        Assume input x has shape (batchsize, n_channels, n_stabilizer, n_charts * height, width)
        """
        assert len(x.shape) == 5, "Data has invalid shape, expected (batchsize, n_channels, n_stabilizer, n_charts * height, width)"
        xs = x.shape
        x = x.view(xs[0], xs[1]*xs[2], 5, -1, xs[-1])  # reshape to individual charts
        x = x[..., 1:-1, 1:-1]  # get rid of padding of the individual charts
        x = x.reshape(xs[0], xs[1]*xs[2], xs[-2] - 2 * 5, xs[-1] - 2)  # reshape back into original shape.
        x = self.down(x)

        # pad the downsized charts.
        xs_new = x.shape
        x = x.view(xs[0], xs[1], xs[2], 5, -1, xs_new[-1])
        x = F.pad(x, (1, 1, 1, 1))  # pad the individual charts
        x = x.view(xs[0], xs[1], xs[2], -1, xs_new[-1] + 2)  # merge the charts, and reshape to original shape (apart from downsampling)

        return x


class Conv2dPadCyclical(nn.Module):
    """
    2d convolution that applies "cylindrical" padding: Zero pad height, cyclical pad width.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, value=0):
        super(Conv2dPadCyclical, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.value = value
        self.padding = padding

    def forward(self, x):
        x = F.pad(x, pad=(self.padding, self.padding, 0, 0), mode="circular")
        x = F.pad(x, pad=(0, 0, self.padding, self.padding), mode="constant", value=self.value)
        x = self.conv(x)
        return x

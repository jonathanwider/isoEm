import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.conv as conv
import math

# open questions: what is rank? --> Probably dimension of data.
# what is with_r? -->


class AddCoords(nn.Module):  # module that adds coord channels.
    def __init__(self, cylindrical=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = 2
        self.cylindrical = cylindrical
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape

        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = torch.arange(dim_y, dtype=torch.int32)

        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)  # why?

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

        if self.cylindrical is False:
            if torch.cuda.is_available() and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)
        else:
            sin_channel = torch.sin(math.pi * yy_channel)
            cos_channel = torch.cos(math.pi * yy_channel)
            if torch.cuda.is_available() and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                sin_channel = sin_channel.cuda()
                cos_channel = cos_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, sin_channel, cos_channel], dim=1)
        return out


class CoordConv(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, use_cuda=True):
        super(CoordConv, self).__init__(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, groups, bias)
        self.addcoords = AddCoords(cylindrical=False, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + 2, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConvCylindrical(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, use_cuda=True, value=0):
        super(CoordConvCylindrical, self).__init__(in_channels, out_channels, kernel_size,
                                        stride, 0, dilation, groups, bias)
        self.value = value
        self.padding = padding
        self.addcoords = AddCoords(cylindrical=True, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + 3, out_channels,
                              kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = F.pad(out, pad=(self.padding, self.padding, 0, 0), mode="circular")
        out = F.pad(out, pad=(0, 0, self.padding, self.padding), mode="constant", value=self.value)
        out = self.conv(out)

        return out

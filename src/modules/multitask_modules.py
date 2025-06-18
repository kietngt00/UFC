from torch import nn
import torch.nn.functional as F
from einops import repeat
import torch


class MultiTaskLinear(nn.Linear):
    def __init__(self, original_linear, n_tasks):
        assert n_tasks > 0

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        super().__init__(in_features, out_features, bias=True)
        
        self.n_tasks = n_tasks
        self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_tasks).contiguous())
        self.original_bias = original_linear.bias.detach() if original_linear.bias is not None else None
        with torch.no_grad():
            self.weight.copy_(original_linear.weight)

    def forward(self, input, t_idx=None):
        output = F.linear(input, self.weight, None)
        if t_idx is not None:
            output = output + self.bias[t_idx][:, None]
        elif self.original_bias is not None:
            output = output + self.original_bias
        return output


class MultiTaskConv2d(nn.Conv2d):
    def __init__(self, original_conv, n_tasks):
        assert n_tasks > 0
        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels
        kernel_size = original_conv.kernel_size
        stride = original_conv.stride
        padding = original_conv.padding
        dilation = original_conv.dilation
        groups = original_conv.groups
        padding_mode = original_conv.padding_mode
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=True, padding_mode=padding_mode)

        self.n_tasks = n_tasks
        self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_tasks).contiguous())
        self.original_bias = original_conv.bias.detach()
        with torch.no_grad():
            self.weight.copy_(original_conv.weight)

    def forward(self, input, t_idx=None):
        output = self._conv_forward(input, self.weight, None)

        if t_idx is not None:
            output = output + self.bias[t_idx][:, :, None, None]
        else:
            output = output + self.original_bias[None, :, None, None]

        return output

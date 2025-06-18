from torch import nn
import torch.nn.functional as F
from einops import repeat
import torch


class MultiTaskLinear(nn.Linear):
    def __init__(self, original_linear, n_tasks=0):
        assert n_tasks > 0

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        super().__init__(in_features, out_features, bias=True)
        
        self.n_tasks = n_tasks
        self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_tasks).contiguous())
        self.original_bias = original_linear.bias.detach()
        with torch.no_grad():
            self.weight.copy_(original_linear.weight)

    def forward(self, input, t_idx=None):
        output = F.linear(input, self.weight, None)
        if t_idx is not None:
            output = output + self.bias[t_idx][:, None]
        else:
            output = output + self.original_bias
        return output

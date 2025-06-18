import torch
import torch.nn as nn

class LoRAConv2d(nn.Module):
    def __init__(self, conv_in, r=4, lora_alpha=1.0):
        super().__init__()
        self.conv = conv_in  # reuse the original conv
        self.r = r
        self.lora_alpha = lora_alpha

        # Add LoRA layers
        self.lora_A = nn.Conv2d(
            conv_in.in_channels, r, conv_in.kernel_size,
            stride=conv_in.stride, padding=conv_in.padding,
            dilation=conv_in.dilation, groups=conv_in.groups, bias=False
        )
        self.lora_B = nn.Conv2d(
            r, conv_in.out_channels, kernel_size=1,
            stride=1, padding=0, bias=False
        )

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        self.lora_A.to(conv_in.weight.dtype)
        self.lora_B.to(conv_in.weight.dtype)

        self.scaling = lora_alpha / r

    def forward(self, x):
        return self.conv(x) + self.scaling * self.lora_B(self.lora_A(x))

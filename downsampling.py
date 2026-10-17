import torch
import torch.nn as nn
from SmartAITool.core import cprint, bprint


class ResidualDownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=4):
        super().__init__()

        padding_val = (kernel_size - 1) // 2
        cprint(padding_val, "yellow")
        
        # Extract features, keep the length
        
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding_val
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_val,
        )

        self.relu = nn.ReLU()


        # Projection for residual connection when dimensions change

        if in_channels != out_channels or stride != 1: 
            self.residual_proj = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size= 1,
            stride=stride,
            padding=0
            )

        else:
            self.residual_proj = nn.Identity()


# torch.manual_seed(0)

# sample_rate = 16000
# duration = 7
# number_samples = duration * sample_rate

# cprint(number_samples)

# batch_size = 2
# x = torch.randn(
#     batch_size, 1, number_samples
# )  ## Fake raw audio batch: (batch, channels, samples)

# print(x.shape)




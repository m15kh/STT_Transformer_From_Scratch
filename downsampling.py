import torch
import torch.nn as nn
from SmartAITool.core import cprint, bprint



class DownsamplingNetwork(nn.Module):
    def __init__(self, embedding_dim=32, hidden_dim = 16, in_channels = 1, initial_mean_pooling_kernel_size = 2, strides = (2, 2)):
        super().__init__()

        self.mean_pooling =  nn.AvgPool1d(
            kernel_size=initial_mean_pooling_kernel_size,
            stride = initial_mean_pooling_kernel_size
        )

        

        self.layers =nn.ModuleList()
        current_in = in_channels 
        for i ,s  in enumerate(strides):

            if i == 0:
                block_in = current_in 
            else:
                block_in = hidden_dim

        self.layers.append(ResidualDownSampleBlock(block_in, hidden_dim, stride=s, kernel_size=8))
        self.final_conv = nn.conv1d(hidden_dim, embedding_dim, kernel_size = 4, padding = "same")

    def forward(self, x):
        x = self.mean_pooling(x)
        cprint(f'after mean_pooling{x.shape}')
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        x = x.transpose(1, 2)
        return x



class ResidualDownSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size=4):
        print("in_channel",in_channels)
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


    def forward(self, x):

        residual = self.residual_proj(x)

        cprint(residual.shape)
        out = self.conv1(x)
        cprint(f"  after conv 1 {out.shape}")
        out = self.bn1(out)
        cprint(f"  after bn1 {out.shape}")
        out = self.relu(out)

        cprint(f"  after relu {out.shape}")
        out = self.conv2(out)

        cprint(f"  after conv2 {out.shape}")

        bprint()
        print(out.shape[-1] , residual.shape[-1])
        bprint()

        if out.shape[-1] != residual.shape[-1]:

            min_len = min(out.shape[-1], residual.shape[-1])
            print(min_len)
            out = out[..., :  min_len]
            residual= residual[..., :min_len]

            bprint()
            print(out.shape[-1] , residual.shape[-1])
            bprint()

#----------------------------------------------------------------------------
        cprint(residual.shape, 'yellow')
        cprint(out.shape,'yellow')
        out = residual + out 

        return self.relu(out)

if __name__ ==  "__main__":

    torch.manual_seed(0)
    block = ResidualDownSampleBlock(in_channels=1, out_channels=800, stride=2)
    block.eval()

    sample_rate = 16000
    duration = 7
    num_samples = sample_rate * duration  # 112000
    batch_size = 2
    x = torch.randn(batch_size, 1, num_samples)

    cprint(f"input:{tuple(x.shape)}",'blue')
AvgPool1d
    with torch.no_grad():
        y = block(x)

    cprint(f"output:{tuple(y.size())}",'blue')

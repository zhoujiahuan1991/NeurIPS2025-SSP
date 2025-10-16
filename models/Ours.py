import torch
import torch.nn as nn

import math


DWCONV3D_DISABLE_CUDNN = True


class InterPrompt(nn.Module):
    def __init__(self, in_channels, spatial_size, out_channels=576):
        super().__init__()
        self.spatial_size = spatial_size

        self.convs = nn.Conv3d(in_channels, in_channels, kernel_size=(1, spatial_size, spatial_size), groups=in_channels)

        self.heads = nn.Linear(in_channels, out_channels)

        nn.init.constant_(self.convs.weight, 0.)
        nn.init.constant_(self.convs.bias, 0.)
        nn.init.xavier_normal_(self.heads.weight)
        nn.init.constant_(self.heads.bias, 0.)


    def forward(self, x):
        B, D, T, H, W = x.shape

        assert H == W == self.spatial_size, f"Spatial size should be {self.spatial_size}, but got {H} and {W}."

        x_conv = self.convs(x) # B, D, T, 1, 1
        x_conv = x_conv.permute(0, 2, 3, 4, 1).reshape(B, T, D)
        output = self.heads(x_conv) # B, T, out_channels

        return output


class Inner_Prompt(nn.Module):

    def __init__(self, in_channels, hidden_channels=384, kernel_size=(1, 1, 1), spatial_size=14):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.conv = nn.Conv3d(
            hidden_channels, hidden_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=hidden_channels,
        )
        self.fc2 = nn.Linear(hidden_channels, in_channels)

        self.inter = InterPrompt(in_channels=hidden_channels, spatial_size=spatial_size, out_channels=in_channels)

        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        B, T, L, C = x.size()

        TL = T * L

        Ca = self.conv.in_channels

        H = W = round(math.sqrt(L))
        assert L == H * W

        x = self.fc1(x)
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous() # (B, Ca, T, H, W)

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x) # (B, Ca, T, H, W)
        inter_prompt = self.inter(x.detach())
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, TL, Ca) # (B T H W Ca) -> (B TL Ca)
        x = self.fc2(x)

        # B T L C
        x = x.reshape(B, T, L, C).contiguous()

        return x, inter_prompt


class PromptAttention(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.down_proj = nn.Linear(dim, self.hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8)
        self.up_proj = nn.Linear(self.hidden_dim, dim)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = self.down_proj(x)
        x = x.permute(1, 0, 2) # [T, B, D]

        attn_output, _ = self.attention(x, x, x)

        attn_output = attn_output.permute(1, 0, 2)
        attn_output = self.up_proj(attn_output)

        scaled_output = attn_output * self.scale
        return scaled_output


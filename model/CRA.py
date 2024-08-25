from typing import List, Any
import torch
import torch.nn as nn
from .Transformer import Transformer
import torch.nn.functional as F


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class CRA(nn.Module):
    def __init__(self, inc, outc, hw, embed_dim, num_patches, depth=4):
        super(CRA, self).__init__()
        self.conv_p1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_p2 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_glb = nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True)

        self.conv_matt = nn.Sequential(nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))
        self.conv_fuse = nn.Sequential(nn.Conv2d(2 * inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))

        self.sigmoid = nn.Sigmoid()
        self.tf = Transformer(depth=depth,
                              num_heads=1,
                              embed_dim=embed_dim,
                              mlp_ratio=3,
                              num_patches=num_patches)
        self.hw = hw
        self.inc = inc

    def forward(self, x, glbmap):
        # x in shape of B,N,C
        # glbmap in shape of B,1,224,224
        B, _, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.hw, self.hw)
        if glbmap.shape[-1] // self.hw != 1:
            glbmap = F.pixel_unshuffle(glbmap, glbmap.shape[-1] // self.hw)
            glbmap = self.conv_glb(glbmap)

        x = torch.cat([glbmap, x], dim=1)
        x = self.conv_fuse(x)
        # pred
        p1 = self.conv_p1(x)
        matt = self.sigmoid(p1)
        matt = matt * (1 - matt)
        matt = self.conv_matt(matt)
        fea = x * (1 + matt)

        # reshape x back to B,N,C
        fea = fea.reshape(B, self.inc, -1).transpose(1, 2)
        fea = self.tf(fea, True)
        p2 = self.conv_p2(fea.transpose(1, 2).reshape(B, C, self.hw, self.hw))

        return [p1, p2, fea]


class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, height, width, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.height = height
        self.width = width

        self.reprojection = nn.Conv2d(value_channels, 2 * value_channels, 1)
        self.norm = nn.LayerNorm(2 * value_channels)

    # x2 should be higher-level representation than x1
    def forward(self, x1, x2):
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)

        # Re-arrange into a (Batch, Embedding dim, Tokens)
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = context.transpose(1, 2) @ query  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, self.height, self.width)
        reprojected_value = self.reprojection(aggregated_values).reshape(B, 2 * D, N).permute(0, 2, 1)
        reprojected_value = self.norm(reprojected_value)

        return reprojected_value


class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """

    def __init__(self, m_dim, in_dim, key_dim, value_dim, hw, head_count=1, token_mlp="mix"):
        super().__init__()
        self.conv_p1 = nn.Conv2d(in_dim, m_dim, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.H = self.W = hw
        self.attn = Cross_Attention(key_dim, value_dim, hw, hw, head_count=head_count)

        self.tf = nn.Sequential(Transformer(depth=1,
                                            num_heads=1,
                                            embed_dim=in_dim * 2,
                                            mlp_ratio=3,
                                            num_patches=hw * hw),
                                Transformer(depth=1,
                                            num_heads=1,
                                            embed_dim=in_dim * 2,
                                            mlp_ratio=3,
                                            num_patches=hw * hw),
                                )

        self.tf2 = nn.Sequential(Transformer(depth=1,
                                             num_heads=1,
                                             embed_dim=in_dim,
                                             mlp_ratio=3,
                                             num_patches=hw * hw),
                                 Transformer(depth=1,
                                             num_heads=1,
                                             embed_dim=in_dim,
                                             mlp_ratio=3,
                                             num_patches=hw * hw),
                                 )

        self.conv_glb = nn.Conv2d(m_dim, in_dim, kernel_size=3, padding=1, bias=True)

        self.conv_matt = nn.Sequential(nn.Conv2d(m_dim, in_dim, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(in_dim),
                                       nn.LeakyReLU(inplace=True))

        self.conv_fuse = nn.Sequential(nn.Conv2d(2 * in_dim, in_dim, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(in_dim),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(in_dim),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(in_dim),
                                       nn.LeakyReLU(inplace=True))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[Any]:
        B, _, _ = x1.size()
        glbmap_conv = self.conv_glb(x2)

        norm_1 = self.norm1(x1)
        norm_2 = self.norm2(glbmap_conv.reshape(B, glbmap_conv.shape[1], -1).transpose(1, 2))

        attn = self.attn(norm_1, norm_2)
        attn = self.tf(attn)

        cra = torch.cat([glbmap_conv.reshape(B, -1, self.H * self.W).transpose(1, 2), x1], dim=2)
        cra = cra + attn
        x = self.conv_fuse(cra.transpose(1, 2).reshape(B, -1, self.H, self.W))
        p1 = self.conv_p1(x)

        fea = self.conv_matt(p1)
        fea = fea.reshape(B, -1, self.H * self.W).transpose(1, 2)
        fea = self.tf2(fea)

        return [p1, fea]

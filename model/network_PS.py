from model.CRA import CrossAttentionBlock
from model.Encoder_pvt import Encoder
from model.Fusion import Up
from model.Transformer import Transformer
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()

        self.encoder = Encoder()

        self.encoder_shaper_8 = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1024), nn.GELU())
        self.encoder_shaper_16 = nn.Sequential(nn.LayerNorm(320), nn.Linear(320, 256), nn.GELU())
        self.encoder_shaper_32 = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU())
        self.encoder_shaper_64 = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 16), nn.GELU())

        self.transformer = nn.ModuleList([Transformer(depth=d,
                                                      num_heads=n,
                                                      embed_dim=e,
                                                      mlp_ratio=m,
                                                      num_patches=p) for d, n, e, m, p in opt.transformer])

        self.p = 8
        self.p2 = 16
        self.p3 = 32
        self.p4 = 64

        self.fuser_8_16 = Up(320, 320, 128, self.p, attn=True)
        self.fuser_16_32 = Up(128, 128, 80, self.p2, attn=True)
        self.fuser_32_64 = Up(64, 64, 32, self.p3, attn=True)

        self.CRA_7 = CrossAttentionBlock(1024, 512, 512, 512, self.p, head_count=2)
        self.CRA_14 = CrossAttentionBlock(256, 320, 320, 320, self.p2, head_count=2)
        self.CRA_28 = CrossAttentionBlock(64, 128, 128, 128, self.p3, head_count=2)
        self.CRA_56 = CrossAttentionBlock(16, 64, 64, 64, self.p4, head_count=2)

    def forward(self, x):
        B = x.shape[0]
        # PVT encoder
        out_8r, out_16r, out_32r, out_64r = self.encoder(x)
        pred = list()

        out_8, out_16, out_32, out_64 = [tf(o) for tf, o, peb in zip(self.transformer,
                                                                          [out_8r, out_16r, out_32r, out_64r],
                                                                          [False, False, False,False])]  # B, patch, feature

        # reshape
        out_8s = self.encoder_shaper_8(out_8) 
        out_8s = out_8s.transpose(1, 2).reshape(B, 1024, self.p, self.p)
        pred.append(out_8s)

        out_16s = self.encoder_shaper_16(out_16)
        out_16s = out_16s.transpose(1, 2).reshape(B, 256, self.p * 2, self.p * 2)
        pred.append(out_16s)

        out_32s = self.encoder_shaper_32(out_32)
        out_32s = out_32s.transpose(1, 2).reshape(B, 64, self.p * 4, self.p * 4)
        pred.append(out_32s)

        out_64s = self.encoder_shaper_64(out_64)
        out_64s = out_64s.transpose(1, 2).reshape(B, 16, self.p * 8, self.p * 8)
        pred.append(out_64s)

        # 8
        p1_8, out_8 = self.CRA_7(out_8, out_8s)
        pred.append(p1_8)

        # 16
        out_16 = self.fuser_8_16(out_16, out_8)
        p1_16, out_16 = self.CRA_14(out_16, out_16s)
        pred.append(p1_16)

        # 32
        out_32 = self.fuser_16_32(out_32, out_16)
        p1_32, out_32 = self.CRA_28(out_32, out_32s)
        pred.append(p1_32)

        # 64
        out_64 = self.fuser_32_64(out_64, out_32)
        p1_64, out_64 = self.CRA_56(out_64, out_64s)
        pred.append(p1_64)

        return pred


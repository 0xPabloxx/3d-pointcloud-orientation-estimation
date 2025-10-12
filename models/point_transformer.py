import torch
import torch.nn as nn

class PointTransformer(nn.Module):
    def __init__(self, in_dim=3, embed_dim=64, num_heads=4, depth=6):
        super().__init__()
        # 这里只示例一个简化版 Point Transformer block
        self.input_proj = nn.Linear(in_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc_out     = nn.Linear(embed_dim, 3)

    def forward(self, x: torch.Tensor):
        # x: (B, N, 3)
        x = self.input_proj(x)        # (B, N, embed_dim)
        x = self.transformer(x)       # (B, N, embed_dim)
        x = x.mean(dim=1)             # 全局平均池化 -> (B, embed_dim)
        return self.fc_out(x)         # (B, 3)

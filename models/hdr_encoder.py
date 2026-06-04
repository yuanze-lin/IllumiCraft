import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class HDRILightEncoder(nn.Module):
    def __init__(
        self,
        num_frames: int = 49,
        in_ch:      int = 3,
        h:          int = 32,
        w:          int = 32,
        mlp_dims:   tuple = (3072, 4096, 4096, 4096, 8192),
        trans_nhead: int   = 8,
        trans_ff:    int   = 2048,
        trans_layers:int   = 1,
        neg_slope: float  = 0.2,
        prompt_dim: int = 4096
    ):
        super().__init__()
        D0, D1, D2, D3, D4 = mlp_dims

        # --- per‑frame MLP as before ---
        self.frame_mlp = nn.Sequential(
            nn.Linear(D0, D1), nn.LeakyReLU(neg_slope, inplace=True),
            nn.Linear(D1, D2), nn.LeakyReLU(neg_slope, inplace=True),
            nn.Linear(D2, D3), nn.LeakyReLU(neg_slope, inplace=True),
            nn.Linear(D3, D4), nn.LeakyReLU(neg_slope, inplace=True),
        )
        self.prompt_dim = prompt_dim
        # --- temporal Transformer as before ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=prompt_dim, nhead=trans_nhead,
            dim_feedforward=trans_ff,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        # --- depth‑wise conv to pool over time ---
        # will take [B*C, prompt_dim, F] → [B*C, prompt_dim, 1]
        self.temporal_conv = nn.Conv1d(
            in_channels=prompt_dim,
            out_channels=prompt_dim,
            kernel_size=num_frames,
            groups=prompt_dim,     # depth‑wise over each dim
            bias=False
        )
        self.act = nn.LeakyReLU(neg_slope, inplace=True)

        self.C = in_ch
        self.F = num_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 49, 32, 32]
        returns: [B, 3, self.prompt_dim]
        """
        B, C, F, H, W = x.shape
        assert C==self.C and F==self.F and H==32 and W==32

        # 1) flatten each frame → [B*F, 3*32*32]
        x = x.permute(0,2,1,3,4).reshape(B*F, C*H*W)

        # 2) per‑frame MLP → [B*F, 2304]
        y = self.frame_mlp(x)
        # 3) view as [B, F, C, self.prompt_dim]
        y = y.view(B, F, -1, self.prompt_dim)

        # 4) permute → [B, 2, F, self.prompt_dim] → merge B,2
        y = y.permute(0,2,1,3).reshape(-1, F, self.prompt_dim)

        # 5) temporal Transformer → [B*C, F, self.prompt_dim]
        y = self.temporal_encoder(y)

        # 6) depth‑wise temporal conv pooling
        #    conv1d wants [N, C, T], so transpose:
        y = y.transpose(1, 2)          # [B*C, self.prompt_dim, F]
        y = self.temporal_conv(y)      # [B*C, self.prompt_dim, 1]
        y = self.act(y)                # non‑linear
        y = y.squeeze(-1)              # [B*C, self.prompt_dim]

        # 7) back to [B, C, self.prompt_dim]
        y = y.view(B, -1, self.prompt_dim)
        return y


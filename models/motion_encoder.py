import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class MotionEncoder(nn.Module):
    def __init__(
        self,
        patch_size=(1, 2, 2),
        in_dim = 16,
        dim = 1536,
        seq_len = 512
    ):
        super().__init__()
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.seq_len = 512

    def forward(self, x: torch.Tensor, seq_len) -> torch.Tensor:
        """
        x: [B, 16, 13, 60, 90]
        returns: [B, 17550, 1536]
        """
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        x = [u.flatten(2).transpose(1, 2) for u in x]
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
              dim=1) for u in x
]           )
        return x
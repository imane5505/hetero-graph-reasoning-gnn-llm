import torch

class DotDecoder(torch.nn.Module):
    def forward(self, z_src, z_dst):
        return (z_src * z_dst).sum(dim=-1)

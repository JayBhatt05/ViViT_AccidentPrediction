#################### Video Vision Transformer(Arnab et al.) Model 2 : Factorized Encoder ###################

import torch
import torch.nn as nn
from CommonModules import create_patches, Encoder


class FactorizedEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, spatial_depth, temporal_depth, max_vehicles, 
                 patch_size=(2, 16, 16)) -> None:
        super(FactorizedEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.max_vehicles = max_vehicles
        self.spatial_encoder = Encoder(embed_dim, num_heads, spatial_depth)
        self.temporal_encoder = Encoder(embed_dim, num_heads, temporal_depth)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.head = nn.Linear(embed_dim, max_vehicles * 2)       #For predicting x and y coordinates of centre
        
    def forward(self, x):
        B = x.size(0)
        x = create_patches(x, self.embed_dim, self.patch_size)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        spatial_tokens = self.spatial_encoder(x)
        temporal_tokens = self.temporal_encoder(spatial_tokens)
        x = self.head(temporal_tokens[:, 0])
        return x.view(B, self.max_vehicles, 2)
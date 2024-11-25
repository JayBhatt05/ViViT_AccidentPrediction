import torch
import torch.nn as nn

class TubeletEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=(5, 16, 16)) -> None:
        super(TubeletEmbedding, self).__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)                     #Convert frames to tubelets by 3D Convolution of 5 frames at a time
        return x.flatten(2).transpose(1, 2)  #Flatten the tubelets to convert them into tokens which will be input to the model
    
    
class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim) -> None:
        super(PositionalEmbedding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, embed_dim))  #Learnable positional embedding for each token
        
    def forward(self, x):
        return x + self.pe
    
    
def create_patches(video, embed_dim, patch_size=(2, 16, 16)):
    B, C, T, H, W = video.size()
    tubelet_embedding = TubeletEmbedding(C, embed_dim, patch_size)
    patches = tubelet_embedding(video)
    seq_len = patches.size(1)
    pos_embedding = PositionalEmbedding(seq_len, embed_dim)
    patches = pos_embedding(patches)
    return patches


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop_rate=0.0) -> None:
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
        self.drop = nn.Dropout(drop_rate)
        
    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x
    
class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, mlp_ratio=0.4, drop_rate=0.0) -> None:
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, mlp_ratio, drop_rate)
                                    for _ in range(depth)])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            
        return x
import os
import numpy as np
import pdb
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

np.random.seed(97)
torch.manual_seed(97)

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, \
                             stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, inputs):
        seq = inputs
        x = self.bn(F.relu(self.cnn(seq)))        
        x = self.dropout(x)
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=batch_first)
    
    def forward(self, inputs):
        seq = inputs
        seq = seq.permute(0, 2, 1)
        x = self.transformer_layer(seq)
        x = x.permute(0, 2, 1)
        
        return x

class MTLucifer(nn.Module):
    def __init__(self, nucleotide_embed_dims=768, nheads=8):
        super().__init__()
        self.nheads = nheads
        self.cls_token_embedding = nn.Parameter(torch.randn(nucleotide_embed_dims))
        self.embed_dims = nucleotide_embed_dims
        
        self.promoter_cnn = nn.Sequential(
                                            CNNBlock(in_channels = 4, out_channels = 256, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            CNNBlock(in_channels = 256, out_channels = 512, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            CNNBlock(in_channels = 512, out_channels = nucleotide_embed_dims, kernel_size = 5, stride = 1, padding = 1, bias=True)
                                         )
        self.promoter_transformer = nn.Sequential(
                                            TransformerBlock(d_model=nucleotide_embed_dims, nhead=self.nheads, batch_first=True),
                                            TransformerBlock(d_model=nucleotide_embed_dims, nhead=self.nheads, batch_first=True),
                                            TransformerBlock(d_model=nucleotide_embed_dims, nhead=self.nheads, batch_first=True),
                                            TransformerBlock(d_model=nucleotide_embed_dims, nhead=self.nheads, batch_first=True),
                                            TransformerBlock(d_model=nucleotide_embed_dims, nhead=self.nheads, batch_first=True)
                                        )
        
    def forward(self, seq):
        seq = seq.permute(0, 2, 1)
        seq = self.promoter_cnn(seq)
        seq = seq.permute(0, 2, 1)
        seq = torch.hstack([self.cls_token_embedding.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], -1, -1), seq]).permute(0, 2, 1)
        outs = self.promoter_transformer(seq)[:, :, 0]

        return outs
    
class PureCNN(nn.Module):
    def __init__(self, embed_dims=1024):
        super().__init__()
        self.embed_dims = embed_dims
        self.promoter_cnn = nn.Sequential(
                                            CNNBlock(in_channels = 4, out_channels = 512, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            CNNBlock(in_channels = 512, out_channels = 768, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            nn.MaxPool1d(5),
                                            CNNBlock(in_channels = 768, out_channels = 768, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            CNNBlock(in_channels = 768, out_channels = 1024, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            nn.MaxPool1d(5)
                                         )
        
        self.linear1 = nn.LazyLinear(2048)
        self.linear2 = nn.Linear(2048, self.embed_dims)
        
    def forward(self, seq):
        seq = seq.permute(0, 2, 1)
        seq = self.promoter_cnn(seq)
        seq = seq.permute(0, 2, 1)
        seq = seq.reshape(seq.shape[0], -1)
        seq1 = self.linear1(seq)
        seq2 = F.relu(seq1)
        outs = self.linear2(seq2)
        return outs

class MotifBasedFCN(nn.Module):
    def __init__(self, num_motifs, motif_embed_dims=512):
        super().__init__()
        self.num_motifs = num_motifs
        self.embed_dims = motif_embed_dims
        
        self.motifs_fcn = nn.Sequential(
                                            nn.Linear(num_motifs, 2048),
                                            nn.ReLU(),
                                            nn.Linear(2048, 1024),
                                            nn.ReLU(),
                                            nn.Linear(1024, 1024),
                                            nn.ReLU(),
                                            nn.Linear(1024, motif_embed_dims)
                                       )
        
    def forward(self, inputs):
        seq = inputs[0]
        motifs = inputs[1]
        
        outs = self.motifs_fcn(motifs)
        
        return outs
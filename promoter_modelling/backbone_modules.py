import os
import numpy as np
import pdb
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from transformers import AutoTokenizer, AutoModel

from tltorch import TRL

np.random.seed(97)
torch.manual_seed(97)

class CNNBlockGN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, \
                             stride=stride, padding=padding, dilation=dilation, bias=bias)
        # self.bn = nn.BatchNorm1d(out_channels)
        self.gn = nn.GroupNorm(out_channels, out_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, inputs):
        seq = inputs
        # x = self.bn(F.relu(self.cnn(seq)))
        x = self.gn(F.relu(self.cnn(seq)))
        x = self.dropout(x)
        
        return x

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
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        stride_for_conv1_and_shortcut = 1

        if in_channels != out_channels:
            stride_for_conv1_and_shortcut = 2

        padding = kernel_size // 2

        # modules for processing the input
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride_for_conv1_and_shortcut, padding = padding, bias=False)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, stride = 1, padding = padding, bias=False)

        # short cut connections
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = stride_for_conv1_and_shortcut, bias=False)

    def forward(self, xl):
        input = xl
        
        if isinstance(self.shortcut, nn.Conv1d):
            input = self.shortcut(xl)
        else:
            input = self.shortcut(input)

        xl = self.relu1(self.bn1(self.conv1(xl)))
        xl = self.conv2(xl)

        xlp1 = input + xl

        xlp1 = self.relu2(self.bn2(xlp1))

        return xlp1

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
    
class MTLuciferGN(nn.Module):
    def __init__(self, nucleotide_embed_dims=768, nheads=8):
        super().__init__()
        self.nheads = nheads
        self.cls_token_embedding = nn.Parameter(torch.randn(nucleotide_embed_dims))
        self.embed_dims = nucleotide_embed_dims
        
        self.promoter_cnn = nn.Sequential(
                                            CNNBlockGN(in_channels = 4, out_channels = 256, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            CNNBlockGN(in_channels = 256, out_channels = 512, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            CNNBlockGN(in_channels = 512, out_channels = nucleotide_embed_dims, kernel_size = 5, stride = 1, padding = 1, bias=True)
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
    
class PureCNNLarge(nn.Module):
    def __init__(self, embed_dims=1024):
        super().__init__()
        self.embed_dims = embed_dims
        self.promoter_cnn = nn.Sequential(
                                            CNNBlock(in_channels = 4, out_channels = 512, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            CNNBlock(in_channels = 512, out_channels = 768, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            nn.MaxPool1d(5),
                                            CNNBlock(in_channels = 768, out_channels = 768, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            CNNBlock(in_channels = 768, out_channels = 1024, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            nn.MaxPool1d(3),
                                            CNNBlock(in_channels = 1024, out_channels = 1024, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            CNNBlock(in_channels = 1024, out_channels = 1024, kernel_size = 5, stride = 1, padding = 1, bias=True),
                                            nn.MaxPool1d(3)
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

class ResNet(nn.Module):
    def __init__(self, embed_dims=1024):
        super().__init__()
        self.embed_dims = embed_dims
        
        self.initial_conv = nn.Conv1d(in_channels=4, out_channels=512, kernel_size=5, stride=1, padding=1, bias=False)
        self.resnet = nn.Sequential(
                                        ResidualBlock(in_channels=512, out_channels=512, kernel_size=5),
                                        ResidualBlock(in_channels=512, out_channels=512, kernel_size=5),
                                        ResidualBlock(in_channels=512, out_channels=768, kernel_size=5),
                                        ResidualBlock(in_channels=768, out_channels=768, kernel_size=5),
                                        ResidualBlock(in_channels=768, out_channels=1024, kernel_size=5),
                                        ResidualBlock(in_channels=1024, out_channels=1024, kernel_size=5),
                                        ResidualBlock(in_channels=1024, out_channels=2048, kernel_size=5),
                                        ResidualBlock(in_channels=2048, out_channels=2048, kernel_size=5)
                                   )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(2048, self.embed_dims)

    def forward(self, seq):
        seq = seq.permute(0, 2, 1)
        seq = self.initial_conv(seq)
        seq = self.resnet(seq)
        seq = self.adaptive_pool(seq)
        seq = seq.reshape(seq.shape[0], -1)
        outs = self.linear(seq)
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
    
class MotifBasedFCNLarge(nn.Module):
    def __init__(self, num_motifs, motif_embed_dims=1024):
        super().__init__()
        self.num_motifs = num_motifs
        self.embed_dims = motif_embed_dims
        
        self.motifs_fcn = nn.Sequential(
                                            nn.Linear(num_motifs, 2048),
                                            nn.ReLU(),
                                            nn.Linear(2048, 4096),
                                            nn.ReLU(),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(),
                                            nn.Linear(4096, 2048),
                                            nn.ReLU(),
                                            nn.Linear(2048, 1024),
                                            nn.ReLU(),
                                            nn.Linear(1024, motif_embed_dims)
                                       )
        
    def forward(self, inputs):
        seq = inputs[0]
        motifs = inputs[1]
        
        outs = self.motifs_fcn(motifs)
        
        return outs
    
class DNABERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("zhihan1996/DNA_bert_6")        
        self.embed_dims = self.model.config.hidden_size
    
    def forward(self, seq):
        outs = self.model(seq)
        return outs.pooler_output

class Concater(nn.Module):
    """
    Concatenates an output of some module with its input alongside some dimension - from https://github.com/autosome-ru/LegNet/blob/main/tutorial/helper_fuctions.py
    Parameters
    ----------
    module : nn.Module
        Module.
    dim : int, optional
        Dimension to concatenate along. The default is -1.
    """
    def __init__(self, module: nn.Module, dim=-1):        
        super().__init__()
        self.mod = module
        self.dim = dim
    
    def forward(self, x):
        return torch.concat((x, self.mod(x)), dim=self.dim)
    
class Bilinear(nn.Module):
    """
    Bilinear layer introduces pairwise product to a NN to model possible combinatorial effects - from https://github.com/autosome-ru/LegNet/blob/main/tutorial/demo_notebook.ipynb
    This particular implementation attempts to leverage the number of parameters via low-rank tensor decompositions.

    Parameters
    ----------
    n : int
        Number of input features.
    out : int, optional
        Number of output features. If None, assumed to be equal to the number of input features. The default is None.
    rank : float, optional
        Fraction of maximal to rank to be used in tensor decomposition. The default is 0.05.
    bias : bool, optional
        If True, bias is used. The default is False.
    """
    def __init__(self, n: int, out=None, rank=0.05, bias=False):        
        super().__init__()
        if out is None:
            out = (n, )
        self.trl = TRL((n, n), out, bias=bias, rank=rank)
        self.trl.weight = self.trl.weight.normal_(std=0.00075)
    
    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        return self.trl(x @ x.transpose(-1, -2))

class SELayer(nn.Module):
    """
    Squeeze-and-Excite layer - from https://github.com/autosome-ru/LegNet/blob/main/tutorial/demo_notebook.ipynb

    Parameters
    ----------
    inp : int
        Middle layer size.
    oup : int
        Input and ouput size.
    reduction : int, optional
        Reduction parameter. The default is 4.
    """
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, int(inp // reduction)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction), int(inp // reduction)),
                Concater(Bilinear(int(inp // reduction), int(inp // reduction // 2), rank=0.5, bias=True)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction) +  int(inp // reduction // 2), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y
    
class LegNetOriginal(nn.Module):
    """
    LegNet neural network - from https://github.com/autosome-ru/LegNet/blob/main/tutorial/demo_notebook.ipynb
    Defaults modified to match the ones in the notebook.

    Parameters
    ----------
    seqsize : int
        Sequence length.
    use_single_channel : bool
        If True, singleton channel is used.
    block_sizes : list, optional
        List containing block sizes. The default is [256, 128, 128, 64, 64, 64, 64].
    ks : int, optional
        Kernel size of convolutional layers. The default is 7.
    resize_factor : int, optional
        Resize factor used in a high-dimensional middle layer of an EffNet-like block. The default is 4.
    activation : nn.Module, optional
        Activation function. The default is nn.SiLU.
    filter_per_group : int, optional
        Number of filters per group in a middle convolutiona layer of an EffNet-like block. The default is 2.
    se_reduction : int, optional
        Reduction number used in SELayer. The default is 4.
    final_ch : int, optional
        Number of channels in the final output convolutional channel. The default is 18.
    bn_momentum : float, optional
        BatchNorm momentum. The default is 0.1.
    """
    __constants__ = ('resize_factor')
    
    def __init__(self, 
                seqsize=250,
                block_sizes=[256, 128, 128, 64, 64, 64, 64], 
                ks=7, 
                resize_factor=4, 
                activation=nn.SiLU,
                filter_per_group=2,
                se_reduction=4,
                final_ch=18,
                bn_momentum=0.1):        
        super().__init__()
        self.block_sizes = block_sizes
        self.resize_factor = resize_factor
        self.se_reduction = se_reduction
        self.seqsize = seqsize
        self.final_ch = final_ch
        self.bn_momentum = bn_momentum
        self.embed_dims = block_sizes[-1] * seqsize

        seqextblocks = OrderedDict()

        in_channels_first_block = 4
        
        block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=in_channels_first_block,
                            out_channels=block_sizes[0],
                            kernel_size=ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(block_sizes[0], momentum=self.bn_momentum),
                       activation()
        )
        seqextblocks[f'blc0'] = block

        
        for ind, (prev_sz, sz) in enumerate(zip(block_sizes[:-1], block_sizes[1:])):
            block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=prev_sz,
                            out_channels=sz * self.resize_factor,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz * self.resize_factor, momentum=self.bn_momentum),
                       activation(),
                       
                       nn.Conv1d(
                            in_channels=sz * self.resize_factor,
                            out_channels=sz * self.resize_factor,
                            kernel_size=ks,
                            groups=sz * self.resize_factor // filter_per_group,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz * self.resize_factor, momentum=self.bn_momentum),
                       activation(),
                
                       SELayer(prev_sz, sz * self.resize_factor, reduction=self.se_reduction),
                
                       nn.Conv1d(
                            in_channels=sz * self.resize_factor,
                            out_channels=prev_sz,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(prev_sz, momentum=self.bn_momentum),
                       activation(),
            
            )
            seqextblocks[f'inv_res_blc{ind}'] = block
            block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=2 * prev_sz,
                            out_channels=sz,
                            kernel_size=ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz, momentum=self.bn_momentum),
                       activation(),
            )
            seqextblocks[f'resize_blc{ind}'] = block

        self.seqextractor = nn.ModuleDict(seqextblocks)

        self.mapper = block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=block_sizes[-1],
                            out_channels=self.final_ch,
                            kernel_size=1,
                            padding='same',
                       ),
                       activation()
        )
        
        self.register_buffer('bins', torch.arange(start=0, end=self.final_ch, step=1, requires_grad=False))
        
    def feature_extractor(self, x):
        x = self.seqextractor['blc0'](x)
        
        for i in range(len(self.block_sizes) - 1):
            x = torch.cat([x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)
        return x 

    def forward(self, x):
        x = x.permute(0, 2, 1)
        f = self.feature_extractor(x)
        return f.reshape(f.shape[0], -1)
    
        # x = self.mapper(f)
        # x = F.adaptive_avg_pool1d(x, 1)
        # x = x.squeeze(2)

        # logprobs = F.log_softmax(x, dim=1) 
        
        # # soft-argmax operation
        # x = F.softmax(x, dim=1)
        # score = (x * self.bins).sum(dim=1)
        
        # return logprobs, score

class LegNet(nn.Module):
    """
    LegNet neural network - from https://github.com/autosome-ru/LegNet/blob/main/tutorial/demo_notebook.ipynb
    Defaults modified to match the ones in the notebook.

    Parameters
    ----------
    seqsize : int
        Sequence length.
    use_single_channel : bool
        If True, singleton channel is used.
    block_sizes : list, optional
        List containing block sizes. The default is [256, 128, 128, 64, 64, 64, 64].
    ks : int, optional
        Kernel size of convolutional layers. The default is 7.
    resize_factor : int, optional
        Resize factor used in a high-dimensional middle layer of an EffNet-like block. The default is 4.
    activation : nn.Module, optional
        Activation function. The default is nn.SiLU.
    filter_per_group : int, optional
        Number of filters per group in a middle convolutiona layer of an EffNet-like block. The default is 2.
    se_reduction : int, optional
        Reduction number used in SELayer. The default is 4.
    final_ch : int, optional
        Number of channels in the final output convolutional channel. The default is 18.
    bn_momentum : float, optional
        BatchNorm momentum. The default is 0.1.
    """
    __constants__ = ('resize_factor')
    
    def __init__(self, 
                seqsize=250,
                block_sizes=[256, 128, 128, 64, 64, 64, 64], 
                ks=7, 
                resize_factor=4, 
                activation=nn.SiLU,
                filter_per_group=2,
                se_reduction=4,
                final_ch=18,
                bn_momentum=0.1):        
        super().__init__()
        self.block_sizes = block_sizes
        self.resize_factor = resize_factor
        self.se_reduction = se_reduction
        self.seqsize = seqsize
        self.final_ch = final_ch
        self.bn_momentum = bn_momentum
        self.embed_dims = block_sizes[-1]

        seqextblocks = OrderedDict()

        in_channels_first_block = 4
        
        block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=in_channels_first_block,
                            out_channels=block_sizes[0],
                            kernel_size=ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(block_sizes[0], momentum=self.bn_momentum),
                       activation()
        )
        seqextblocks[f'blc0'] = block

        
        for ind, (prev_sz, sz) in enumerate(zip(block_sizes[:-1], block_sizes[1:])):
            block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=prev_sz,
                            out_channels=sz * self.resize_factor,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz * self.resize_factor, momentum=self.bn_momentum),
                       activation(),
                       
                       nn.Conv1d(
                            in_channels=sz * self.resize_factor,
                            out_channels=sz * self.resize_factor,
                            kernel_size=ks,
                            groups=sz * self.resize_factor // filter_per_group,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz * self.resize_factor, momentum=self.bn_momentum),
                       activation(),
                
                       SELayer(prev_sz, sz * self.resize_factor, reduction=self.se_reduction),
                
                       nn.Conv1d(
                            in_channels=sz * self.resize_factor,
                            out_channels=prev_sz,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(prev_sz, momentum=self.bn_momentum),
                       activation(),
            
            )
            seqextblocks[f'inv_res_blc{ind}'] = block
            block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=2 * prev_sz,
                            out_channels=sz,
                            kernel_size=ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz, momentum=self.bn_momentum),
                       activation(),
            )
            seqextblocks[f'resize_blc{ind}'] = block

        self.seqextractor = nn.ModuleDict(seqextblocks)

        self.mapper = block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=block_sizes[-1],
                            out_channels=self.final_ch,
                            kernel_size=1,
                            padding='same',
                       ),
                       activation()
        )
        
        self.register_buffer('bins', torch.arange(start=0, end=self.final_ch, step=1, requires_grad=False))
        
    def feature_extractor(self, x):
        x = self.seqextractor['blc0'](x)
        
        for i in range(len(self.block_sizes) - 1):
            x = torch.cat([x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)
        return x 

    def forward(self, x):
        # pdb.set_trace()
        x = x.permute(0, 2, 1)
        f = self.feature_extractor(x)
    
        # x = self.mapper(f)
        x = F.adaptive_avg_pool1d(f, 1)
        x = x.squeeze(2)

        return x

        # logprobs = F.log_softmax(x, dim=1) 
        
        # # soft-argmax operation
        # x = F.softmax(x, dim=1)
        # score = (x * self.bins).sum(dim=1)
        
        # return logprobs, score

class LegNetLarge(nn.Module):
    """
    LegNet neural network - from https://github.com/autosome-ru/LegNet/blob/main/tutorial/demo_notebook.ipynb

    Parameters
    ----------
    seqsize : int
        Sequence length.
    use_single_channel : bool
        If True, singleton channel is used.
    block_sizes : list, optional
        List containing block sizes. The default is [256, 128, 128, 64, 64, 64, 64].
    ks : int, optional
        Kernel size of convolutional layers. The default is 7.
    resize_factor : int, optional
        Resize factor used in a high-dimensional middle layer of an EffNet-like block. The default is 4.
    activation : nn.Module, optional
        Activation function. The default is nn.SiLU.
    filter_per_group : int, optional
        Number of filters per group in a middle convolutiona layer of an EffNet-like block. The default is 2.
    se_reduction : int, optional
        Reduction number used in SELayer. The default is 4.
    final_ch : int, optional
        Number of channels in the final output convolutional channel. The default is 18.
    bn_momentum : float, optional
        BatchNorm momentum. The default is 0.1.
    """
    __constants__ = ('resize_factor')
    
    def __init__(self, 
                seqsize=250,
                block_sizes=[1024, 512, 512, 256, 256, 256, 256], 
                ks=7, 
                resize_factor=4, 
                activation=nn.SiLU,
                filter_per_group=2,
                se_reduction=4,
                final_ch=18,
                bn_momentum=0.1):        
        super().__init__()
        self.block_sizes = block_sizes
        self.resize_factor = resize_factor
        self.se_reduction = se_reduction
        self.seqsize = seqsize
        self.final_ch = final_ch
        self.bn_momentum = bn_momentum
        self.embed_dims = block_sizes[-1]

        seqextblocks = OrderedDict()

        in_channels_first_block = 4
        
        block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=in_channels_first_block,
                            out_channels=block_sizes[0],
                            kernel_size=ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(block_sizes[0], momentum=self.bn_momentum),
                       activation()
        )
        seqextblocks[f'blc0'] = block

        
        for ind, (prev_sz, sz) in enumerate(zip(block_sizes[:-1], block_sizes[1:])):
            block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=prev_sz,
                            out_channels=sz * self.resize_factor,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz * self.resize_factor, momentum=self.bn_momentum),
                       activation(),
                       
                       nn.Conv1d(
                            in_channels=sz * self.resize_factor,
                            out_channels=sz * self.resize_factor,
                            kernel_size=ks,
                            groups=sz * self.resize_factor // filter_per_group,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz * self.resize_factor, momentum=self.bn_momentum),
                       activation(),
                
                       SELayer(prev_sz, sz * self.resize_factor, reduction=self.se_reduction),
                
                       nn.Conv1d(
                            in_channels=sz * self.resize_factor,
                            out_channels=prev_sz,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(prev_sz, momentum=self.bn_momentum),
                       activation(),
            
            )
            seqextblocks[f'inv_res_blc{ind}'] = block
            block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=2 * prev_sz,
                            out_channels=sz,
                            kernel_size=ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz, momentum=self.bn_momentum),
                       activation(),
            )
            seqextblocks[f'resize_blc{ind}'] = block

        self.seqextractor = nn.ModuleDict(seqextblocks)

        self.mapper = block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=block_sizes[-1],
                            out_channels=self.final_ch,
                            kernel_size=1,
                            padding='same',
                       ),
                       activation()
        )
        
        self.register_buffer('bins', torch.arange(start=0, end=self.final_ch, step=1, requires_grad=False))
        
    def feature_extractor(self, x):
        x = self.seqextractor['blc0'](x)
        
        for i in range(len(self.block_sizes) - 1):
            x = torch.cat([x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)
        return x 

    def forward(self, x):
        # pdb.set_trace()
        x = x.permute(0, 2, 1)
        f = self.feature_extractor(x)
    
        # x = self.mapper(f)
        x = F.adaptive_avg_pool1d(f, 1)
        x = x.squeeze(2)

        return x

        # logprobs = F.log_softmax(x, dim=1) 
        
        # # soft-argmax operation
        # x = F.softmax(x, dim=1)
        # score = (x * self.bins).sum(dim=1)
        
        # return logprobs, score
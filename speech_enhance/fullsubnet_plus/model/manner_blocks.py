import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GTU1d(nn.Module):
    """
    Gated Tanh Units for 1D inputs
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        """
        Args:
            in_channels <int>
            out_channels <int>
        """
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels, self.out_channels = in_channels, out_channels

        self.map = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.map_gate = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, T)
        Returns:
            output (batch_size, out_channels, T)
        """
        x_output = self.map(input)
        x_output = torch.tanh(x_output)
        x_gate = self.map_gate(input)
        x_gate = torch.sigmoid(x_gate)
        
        output = x_output * x_gate
        
        return output

class SinglePathProcessing(nn.Module):
    """
    Chunking without overlapp.
    segment_len: chunk size
    """
    def __init__(self, segment_len):
        super().__init__()
        
        self.segment_len = segment_len
        
    def pad_segment(self, input):
        """This pad_segment is for direct segment pad without stride"""
        # input size : (B, N, T)
        b, dim, s = input.shape
        rest      = s % self.segment_len
        
        if rest > 0 :
            rest  = self.segment_len - rest
            pad   = (torch.zeros(b, dim, rest)).type(input.type()).to(input.device)
            input = torch.cat([input, pad], dim = -1)
            
        return input, rest
    
    def segmentation(self, input):
        """This segment is for direct segment without stride"""
        # (B, N, T)
        input, rest = self.pad_segment(input)
        b, dim, s   = input.shape
        
        # (B, N, L, T)
        segments = input.view(b, dim, -1,  self.segment_len)
        
        return segments, rest
    
class DualPathProcessing(nn.Module):

    # -*- coding: utf-8 -*-
    # Original copyright:
    # Asteroid (https://github.com/asteroid-team/asteroid)
    """
    Overlapped chunking.
    chunk_size: chunk size.
    hop_size: overlapping size.
    """

    def __init__(self, chunk_size, hop_size):
        super(DualPathProcessing, self).__init__()
        self.chunk_size    = chunk_size
        self.hop_size      = hop_size
        self.n_orig_frames = None

    def unfold(self, x):
        
        # x is (batch, chan, frames)
        batch, chan, frames = x.size()
        assert x.ndim == 3
        self.n_orig_frames = x.shape[-1]
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        return unfolded.reshape(
            batch, chan, self.chunk_size, -1
        )  # (batch, chan, chunk_size, n_chunks)

    def fold(self, x, output_size=None):
        r"""
        Folds back the spliced feature tensor.
        Input shape $(batch, channels, chunksize, nchunks)$ to original shape
        $(batch, channels, time)$ using overlap-add.
        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                $(batch, channels, chunksize, nchunks)$.
            output_size (int, optional): sequence length of original feature tensor.
                If None, the original length cached by the previous call of
                :meth:`unfold` will be used.
        Returns:
            :class:`torch.Tensor`:  feature tensor of shape $(batch, channels, time)$.
        .. note:: `fold` caches the original length of the input.
        """
        output_size = output_size if output_size is not None else self.n_orig_frames
        # x is (batch, chan, chunk_size, n_chunks)
        batch, chan, chunk_size, n_chunks = x.size()
        to_unfold = x.reshape(batch, chan * self.chunk_size, n_chunks)
        x = torch.nn.functional.fold(
            to_unfold,
            (output_size, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        # force float div for torch jit
        x /= float(self.chunk_size) / self.hop_size

        return x.reshape(batch, chan, self.n_orig_frames)

    @staticmethod
    def intra_process(x, module):
        r"""Performs intra-chunk processing.
        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                (batch, channels, chunk_size, n_chunks).
            module (:class:`torch.nn.Module`): module one wish to apply to each chunk
                of the spliced feature tensor.
        Returns:
            :class:`torch.Tensor`: processed spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.
        .. note:: the module should have the channel first convention and accept
            a 3D tensor of shape $(batch, channels, time)$.
        """

        # x is (batch, channels, chunk_size, n_chunks)
        batch, channels, chunk_size, n_chunks = x.size()
        # we reshape to batch*chunk_size, channels, n_chunks
        x = x.transpose(1, -1).reshape(batch * n_chunks, chunk_size, channels).transpose(1, -1)
        x = module(x)
        x = x.reshape(batch, n_chunks, channels, chunk_size).transpose(1, -1).transpose(1, 2)
        return x

    @staticmethod
    def inter_process(x, module):
        r"""Performs inter-chunk processing.
        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                $(batch, channels, chunksize, nchunks)$.
            module (:class:`torch.nn.Module`): module one wish to apply between
                each chunk of the spliced feature tensor.
        Returns:
            x (:class:`torch.Tensor`): processed spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.
        .. note:: the module should have the channel first convention and accept
            a 3D tensor of shape $(batch, channels, time)$.
        """

        batch, channels, chunk_size, n_chunks = x.size()
        x = x.transpose(1, 2).reshape(batch * chunk_size, channels, n_chunks)
        x = module(x)
        x = x.reshape(batch, chunk_size, channels, n_chunks).transpose(1, 2)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
            
        return x
    
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        return x * self.sigmoid(x)

class DepthwiseConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False, dilation=1):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                              groups=in_channels, stride=stride, padding=padding, bias=bias, dilation=dilation)

    def forward(self, x):
        
        x = self.conv(x)
        
        return x
    
class PointwiseConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, padding=0, bias=True):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        
        x = self.conv(x)
        
        return x

class Transpose(nn.Module):
    
    def __init__(self, p1, p2):
        super(Transpose, self).__init__()
        self.p1 = p1
        self.p2 = p2

    def forward(self, x):
        return x.transpose(self.p1, self.p2)

class ResConBlock(nn.Module):
    """
    Residual Conformer block.
        in_channels  :  in channel in encoder and decoder.
        kernel_size  :  kernel size for depthwise convolution.
        growth1      :  expanding channel size and reduce after GLU.
        growth2      :  decide final channel size in the block, encoder for 2, decoder for 1/2.
    """
    def __init__(self, in_channels, kernel_size=31, growth1=2, growth2=2, causal=False, dilation=1, favor_attention=False):
        super().__init__()
        
        out_channels1 = int(in_channels*growth1)
        out_channels2 = int(in_channels*growth2)

        print(f'ResConBlock dilation: {dilation}')
        
        self.point_conv1 = PointwiseConv(in_channels, out_channels1, stride=1, padding=0, bias=True)
        
        self.causal = causal
        self.favor_attention = favor_attention
        if self.favor_attention:
            # self.att = SelfAttention(dim = out_channels1, heads = 1)
            self.att = GlobalAttention(1, out_channels1, out_channels1, out_channels1)
            # self.att = WindowAttention(dim = out_channels1, window_size = 100, autopad=True)

            self.GLUs = nn.Sequential(nn.BatchNorm1d(out_channels1),
                                        PointwiseConv(out_channels1, out_channels1, stride=1, padding=0, bias=True),
                                        nn.GLU(dim=1))
        else:
            self.GLUs = nn.Sequential(nn.BatchNorm1d(out_channels1), 
                                        nn.GLU(dim=1))

        assert kernel_size % 2 == 1, 'kernel_size {kernel_size} must be odd.'
        self.depth_padding = dilation * (kernel_size - 1) if causal \
                                else (dilation * (kernel_size - 1)) // 2
        self.depth_conv  = nn.Sequential(
                                DepthwiseConv(in_channels, in_channels, kernel_size, stride=1,
                                                padding=self.depth_padding, dilation=dilation),
                                nn.BatchNorm1d(in_channels), Swish())
        self.point_conv2 = nn.Sequential(
                                PointwiseConv(in_channels, out_channels2, stride=1, padding=0, bias=True),
                                nn.BatchNorm1d(out_channels2), Swish())
        self.conv     = BasicConv(out_channels2, out_channels2, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_channels, out_channels2, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        
        out = self.point_conv1(x)
        
        if self.favor_attention:
            out = out.transpose(-1, -2)
            out += self.att(out, out, out)
            out = out.transpose(-1, -2)
        
        out = self.GLUs(out)
        
        out = self.depth_conv(out)

        if self.causal:
            out = out[:, :, :-(self.depth_padding)]
        
        out = self.point_conv2(out)
        out = self.conv(out)
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out

class Expendable_ResConBlock(nn.Module):
    """
    Residual Conformer block.
        in_channels  :  in channel in encoder and decoder.
        kernel_size  :  kernel size for depthwise convolution.
        growth1      :  expanding channel size and reduce after GLU.
        growth2      :  decide final channel size in the block, encoder for 2, decoder for 1/2.
    """
    def __init__(self, in_channels, kernel_size=31, growth1=2, growth2=2, causal=False, dilation=1, favor_attention=False, GLU_growth=2):
        super().__init__()
        
        out_channels1 = int(in_channels*growth1)
        out_channels2 = int(in_channels*growth2)

        print(f'ResConBlock dilation: {dilation}')
        # GLU_growth default to 2
        channel_before_glu = out_channels1 * GLU_growth
        self.point_conv1 = PointwiseConv(in_channels, channel_before_glu, stride=1, padding=0, bias=True)
        
        self.causal = causal
        self.favor_attention = favor_attention
        if self.favor_attention:
            self.att = GlobalAttention(1, channel_before_glu, channel_before_glu, channel_before_glu)

            self.GLUs = nn.Sequential(nn.BatchNorm1d(channel_before_glu),
                                        PointwiseConv(channel_before_glu, channel_before_glu, stride=1, padding=0, bias=True),
                                        nn.GLU(dim=1))
        else:
            self.GLUs = nn.Sequential(nn.BatchNorm1d(channel_before_glu), 
                                        nn.GLU(dim=1))

        assert kernel_size % 2 == 1, 'kernel_size {kernel_size} must be odd.'
        self.depth_padding = dilation * (kernel_size - 1) if causal \
                                else (dilation * (kernel_size - 1)) // 2
        self.depth_conv  = nn.Sequential(
                                DepthwiseConv(out_channels1, out_channels1, kernel_size, stride=1,
                                                padding=self.depth_padding, dilation=dilation),
                                nn.BatchNorm1d(out_channels1), Swish())
        self.point_conv2 = nn.Sequential(
                                PointwiseConv(out_channels1, out_channels2, stride=1, padding=0, bias=True),
                                nn.BatchNorm1d(out_channels2), Swish())
        self.conv     = BasicConv(out_channels2, out_channels2, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_channels, out_channels2, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        
        out = self.point_conv1(x)
        if self.favor_attention:
            out = out.transpose(-1, -2)
            out_att = self.att(out, out, out)
            out = out + out_att
            out = out.transpose(-1, -2)
        
        out = self.GLUs(out)
        
        out = self.depth_conv(out)

        if self.causal and self.depth_padding != 0:
            out = out[:, :, :-(self.depth_padding)]
        
        out = self.point_conv2(out)
        out = self.conv(out)
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
            
        return x

class DepthwiseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                              groups=in_channels, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        
        x = self.conv(x)
        
        return x
    
class PointwiseConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        
        x = self.conv(x)
        
        return x

class ResConBlock2d_Causal(nn.Module):
    """
    Residual Conformer block.
        in_channels  :  in channel in encoder and decoder.
        kernel_size  :  kernel size for depthwise convolution.
        growth1      :  expanding channel size and reduce after GLU.
        growth2      :  decide final channel size in the block, encoder for 2, decoder for 1/2.
    """
    def __init__(self, in_channels, kernel_size=31, growth1=2, growth2=2):
        super().__init__()
        
        out_channels1 = int(in_channels*growth1)
        out_channels2 = int(in_channels*growth1 // 2)
        
        self.point_conv1 = nn.Sequential(
                                PointwiseConv2d(in_channels, out_channels1, stride=1, padding=0, bias=True),
                                nn.BatchNorm2d(out_channels1), nn.GLU(dim=1))
        self.depth_conv  = nn.Sequential(
                                DepthwiseConv2d(out_channels2, out_channels2, (kernel_size, 1), stride=1, padding=((kernel_size - 1) // 2, 0)),
                                nn.BatchNorm2d(out_channels2), Swish())
        self.point_conv2 = nn.Sequential(
                                PointwiseConv2d(out_channels2, out_channels2, stride=1, padding=0, bias=True),
                                nn.BatchNorm2d(out_channels2), Swish())
        self.conv     = BasicConv2d(out_channels2, in_channels, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv2d(in_channels, in_channels, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        
        out = self.point_conv1(x)
        out = self.depth_conv(out)
        out = self.point_conv2(out)
        out = self.conv(out)
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out

class ResConBlock2d(nn.Module):
    """
    Residual Conformer block.
        in_channels  :  in channel in encoder and decoder.
        kernel_size  :  kernel size for depthwise convolution.
        growth1      :  expanding channel size and reduce after GLU.
        growth2      :  decide final channel size in the block, encoder for 2, decoder for 1/2.
    """
    def __init__(self, in_channels, kernel_size=31, growth1=2, growth2=2):
        super().__init__()
        
        out_channels1 = int(in_channels*growth1)
        out_channels2 = int(in_channels*growth2)
        
        self.point_conv1 = nn.Sequential(
                                PointwiseConv2d(in_channels, out_channels1, stride=1, padding=0, bias=True),
                                nn.BatchNorm2d(out_channels1), nn.GLU(dim=1))
        self.depth_conv  = nn.Sequential(
                                DepthwiseConv2d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                                nn.BatchNorm2d(in_channels), Swish())
        self.point_conv2 = nn.Sequential(
                                PointwiseConv2d(in_channels, out_channels2, stride=1, padding=0, bias=True),
                                nn.BatchNorm2d(out_channels2), Swish())
        self.conv     = BasicConv2d(out_channels2, out_channels2, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv2d(in_channels, out_channels2, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        
        out = self.point_conv1(x)
        out = self.depth_conv(out)
        out = self.point_conv2(out)
        out = self.conv(out)
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out

class ChannelAttention(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.fc = nn.Sequential(nn.Linear(channels, channels//2), nn.ReLU(),
                               nn.Linear(channels//2, channels))        
        
    def forward(self, x):
        """
        Input X: [B,N,T]
        Ouput X: [B,N,T]
        """
        
        # [B,N,T] -> [B,N,1]
        attn_max = F.adaptive_max_pool1d(x, 1)
        attn_avg = F.adaptive_avg_pool1d(x, 1)
        
        attn_max = self.fc(attn_max.squeeze())
        attn_avg = self.fc(attn_avg.squeeze())
        
        # [B,N,1]
        attn = attn_max + attn_avg
        attn = torch.sigmoid(attn).unsqueeze(-1)
        
        # [B,N,T]
        x = x * attn
        
        return x 
    
class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, temperature):
        super().__init__()
        
        self.temperature = temperature

    def forward(self, q, k, v):

        attn   = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn   = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn

class GlobalAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        # d_model = dimension of model = features
        self.n_head = n_head
        self.d_k    = d_k
        self.d_v    = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc   = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, q, k, v):
        
        """
        Input X: [B*N/3,P,C]
        Output X: [B*N/3,P,C]
        """
        # [B*N,P,C]
        b, p, c     = q.shape 
        d_k, n_head = self.d_k, self.n_head

        # [B*N,P,C] -> [B*N,P,N_head,D_k]
        q = self.w_qs(q).view(b, p, n_head, d_k) 
        k = self.w_ks(k).view(b, p, n_head, d_k)
        v = self.w_vs(v).view(b, p, n_head, d_k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, attn = self.attention(q, k, v)

        q = q.transpose(1, 2).contiguous().view(b, p, -1)
        q = self.fc(q)

        return q

class LocalAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        kernel_size1    = 31
        kernel_size2    = 7
        self.depth_conv = nn.Sequential(DepthwiseConv(channels, channels, kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2),
                                        nn.BatchNorm1d(channels), Swish()) 
        self.conv       = BasicConv(2, 1, kernel_size2, stride=1, padding=(kernel_size2-1) // 2, relu=False)
        
    def forward(self, x):
        """
        Input X: [B,N/3,P,C]
        Output X: [B,N/3,P,C]
        """

        b, n, p, c = x.size()
        # [B,N/3,P,C] -> [B*P,N/3,C]
        attn = x.permute(0,2,1,3).contiguous().view(b*p, n, c)
        attn = self.depth_conv(attn)
        attn = torch.cat([torch.max(attn, dim=1)[0].unsqueeze(1), torch.mean(attn, dim=1).unsqueeze(1)], dim=1)
        attn = self.conv(attn)
        # [B*P,1,C]
        attn = torch.sigmoid(attn)
        attn = attn.view(b, p, 1, c).permute(0,2,1,3).contiguous()
        x = x * attn    

        return x

class MultiviewAttentionBlock(nn.Module):
    """
    Multiview Attention block.
        channels     :  in channel in encoder.
        head         :  number of heads in global attention.
        segment_len  :  chunk size for overlapped chunking in global and local attention.
    """
    def __init__(self, channels, segment_len, head):
        super().__init__()
        
        self.inter = int(channels / 3)
        d_k        = int(segment_len * head)
        
        self.dsp   = DualPathProcessing(segment_len, segment_len//2)
        
        self.in_branch0 = BasicConv(channels, self.inter, kernel_size=1, stride=1)
        self.in_branch1 = BasicConv(channels, self.inter, kernel_size=1, stride=1)
        self.in_branch2 = BasicConv(channels, self.inter, kernel_size=1, stride=1)
        
        self.channel_attn = ChannelAttention(self.inter)
        self.global_attn  = GlobalAttention(head, segment_len, d_k, d_k)
        self.local_attn   = LocalAttention(self.inter)
        
        self.out_branch0 = BasicConv(self.inter, self.inter, kernel_size=3, stride=1, padding=1)
        self.out_branch1 = BasicConv(self.inter, self.inter, kernel_size=3, stride=1, padding=1)
        self.out_branch2 = BasicConv(self.inter, self.inter, kernel_size=3, stride=1, padding=1)

        self.conv     = BasicConv(self.inter*3, channels, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(channels, channels, kernel_size=1, stride=1, relu=False)
        
        # mask gate as activation unit
        self.output_tanh    = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Tanh())
        self.output_sigmoid = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Sigmoid())
        self.gate_conv      = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.ReLU())
        
    def forward(self, x):
        """
        Input X: [B,N,T] pass through three-path with x:[B,N/3,T], respectively.
        Ouput X: [B,N,T]
        C: chunk size
        P: number of chunks
        {x0: channel path, x1: global path, x2: local path}
        """
        # [B,N,T] -> [B,N/3,T]
        x0 = self.in_branch0(x)
        x1 = self.in_branch1(x)
        x2 = self.in_branch2(x)
        
        # overlapped chunking / (B,N/3,C,P) -> [B,N/3,P,C]
        x1 = self.dsp.unfold(x1).transpose(2,3)
        x2 = self.dsp.unfold(x2).transpose(2,3)
        
        b,n,p,c = x1.size()
        
        # [B*N/3,P,C]
        x1 = x1.view(b*n,p,c) 
        
        x0 = self.channel_attn(x0)
        x1 = self.global_attn(x1, x1, x1)
        x2 = self.local_attn(x2)
        
        x1 = x1.view(b,n,p,c)
        
        # [B,N/3,P,C] -> [B,N/3,T]
        x1 = self.dsp.fold(x1.transpose(2,3))
        x2 = self.dsp.fold(x2.transpose(2,3))
        
        x0 = self.out_branch0(x0)
        x1 = self.out_branch1(x1)
        x2 = self.out_branch2(x2)

        # Concat: [B,N/3,T]*3 -> [B,N,T]
        out   = torch.cat([x0, x1, x2], dim=1)
        out   = self.conv(out)
        short = self.shortcut(x)
        
        # mask gate
        gated_tanh = self.output_tanh(out)
        gated_sig  = self.output_sigmoid(out)
        gated      = gated_tanh * gated_sig
        out        = self.gate_conv(gated)
        
        x = short + out
     
        return x

class ChannelAttention2d(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.fc = nn.Sequential(nn.Linear(channels, channels//2), nn.ReLU(),
                               nn.Linear(channels//2, channels))        
        
    def forward(self, x):
        """
        Input X: [B, F, SB, T]
        Ouput X: [B,N,T]
        """
        
        # [B, F, SB, T] -> [B,N,1]
        attn_max = F.adaptive_max_pool2d(x, (None, 1))
        attn_avg = F.adaptive_avg_pool2d(x, (None, 1))
        # print(self.fc, attn_max.shape)
        attn_max = self.fc(attn_max.squeeze())
        attn_avg = self.fc(attn_avg.squeeze())
        
        # [B,N,1]
        attn = attn_max + attn_avg
        attn = torch.sigmoid(attn).unsqueeze(-1)
        
        # [B,N,T]
        x = x * attn
        
        return x 

class MultiviewAttentionBlock2d(nn.Module):
    """
    Multiview Attention block.
        channels     :  in channel in encoder.
        head         :  number of heads in global attention.
        segment_len  :  chunk size for overlapped chunking in global and local attention.
    """
    def __init__(self, channels, segment_len, head):
        super().__init__()
        
        self.inter = int(channels / 3)
        d_k        = int(segment_len * head)
        # segment_len = sub-band size
        # self.dsp   = DualPathProcessing(segment_len, segment_len//2)
        
        self.in_branch0 = BasicConv2d(channels, self.inter, kernel_size=1, stride=1)
        self.in_branch1 = BasicConv2d(channels, self.inter, kernel_size=1, stride=1)
        self.in_branch2 = BasicConv2d(channels, self.inter, kernel_size=1, stride=1)
        
        self.channel_attn = ChannelAttention2d(segment_len)
        self.global_attn  = GlobalAttention(head, segment_len, d_k, d_k)
        self.local_attn   = LocalAttention(self.inter)
        
        self.out_branch0 = BasicConv2d(self.inter, self.inter, kernel_size=3, stride=1, padding=1)
        self.out_branch1 = BasicConv2d(self.inter, self.inter, kernel_size=3, stride=1, padding=1)
        self.out_branch2 = BasicConv2d(self.inter, self.inter, kernel_size=3, stride=1, padding=1)

        self.conv     = BasicConv2d(self.inter*3, channels, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv2d(channels, channels, kernel_size=1, stride=1, relu=False)
        
        # mask gate as activation unit
        self.output_tanh    = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.Tanh())
        self.output_sigmoid = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.Sigmoid())
        self.gate_conv      = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.ReLU())
        
    def forward(self, x):
        """
        Input X: [B, F, SB, T]
        Ouput X: [B,N,T]
        C: chunk size
        P: number of chunks
        {x0: channel path, x1: global path, x2: local path}
        """
        # [B, F, SB, T] -> [B, F/3, SB, T]
        x0 = self.in_branch0(x)
        x1 = self.in_branch1(x)
        x2 = self.in_branch2(x)
        
        # overlapped chunking / (B,N/3,C,P) -> [B,N/3,P,C]
        # x1 = self.dsp.unfold(x1).transpose(2,3)
        # x2 = self.dsp.unfold(x2).transpose(2,3)

        # [B, F/3, SB, T] -> [B, F/3, T, SB]
        x1 = x1.transpose(2,3)
        x2 = x2.transpose(2,3)
        
        b,n,p,c = x1.size()
        
        # [B*N/3,P,C]
        x1 = x1.view(b*n,p,c) 

        x0 = self.channel_attn(x0) # [B, F/3, SB, T]
        x1 = self.global_attn(x1, x1, x1)
        x2 = self.local_attn(x2)
        
        x1 = x1.view(b,n,p,c)
        
        # [B,N/3,P,C] -> [B,N/3,T]
        # x1 = self.dsp.fold(x1.transpose(2,3))
        # x2 = self.dsp.fold(x2.transpose(2,3))
        x1 = x1.transpose(2,3)
        x2 = x2.transpose(2,3)
        
        x0 = self.out_branch0(x0)
        x1 = self.out_branch1(x1)
        x2 = self.out_branch2(x2)

        # Concat: [B,N/3,T]*3 -> [B,N,T]
        out   = torch.cat([x0, x1, x2], dim=1)
        out   = self.conv(out)
        short = self.shortcut(x)
        
        # mask gate
        gated_tanh = self.output_tanh(out)
        gated_sig  = self.output_sigmoid(out)
        gated      = gated_tanh * gated_sig
        out        = self.gate_conv(gated)
        
        x = short + out
     
        return x

class Encoder(nn.Module):
  
    def __init__(self, in_channels, out_channels, kernel_size, stride, segment_len, head):
        super().__init__()
        
        self.down_conv  = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size, stride),
                                        nn.BatchNorm1d(in_channels), nn.ReLU())
        self.conv_block = ResConBlock(in_channels, growth1=2, growth2=2)
        self.attn_block = MultiviewAttentionBlock(out_channels, segment_len, head)

    def forward(self, x):

        x = self.down_conv(x)        
        x = self.conv_block(x)
        x = self.attn_block(x)

        return x

class Encoder2D(nn.Module):
  
    def __init__(self, in_channels, out_channels, kernel_size, stride, subband, head):
        super().__init__()
        self.depth=1
        self.kernel_size, self.stride = kernel_size, stride
        self.down_conv  = nn.Sequential(nn.Conv2d(in_channels, in_channels, (kernel_size, 1), (stride, 1)),
                                        nn.BatchNorm2d(in_channels), nn.ReLU())
        self.conv_block = ResConBlock2d(in_channels, growth1=2, growth2=1)
        self.attn_block = MultiviewAttentionBlock2d(out_channels, subband, head)

    def padding(self, length):

        length = math.ceil(length)
        length = math.ceil((length - self.kernel_size) / self.stride) + 1
        length = max(length, 1)
        length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        
        return int(length)
    
    def forward(self, x):
        """
            Input: [bn, feat, subband, time]
        """
        bn, feat, subband, time = x.size()

        padding = self.padding(subband)
        x = F.pad(x, (0, 0, 0 , padding - subband))
        x = self.down_conv(x)        
        x = self.conv_block(x)
        x = self.attn_block(x)

        return x

class Encoder2D_submain(nn.Module):
  
    def __init__(self, in_channels, out_channels, kernel_size, stride, subband, head, attention=True):
        super().__init__()
        self.depth=1
        self.kernel_size, self.stride = kernel_size, stride
        self.down_conv  = nn.Sequential(nn.Conv2d(in_channels, in_channels, (kernel_size, 1), (stride, 1)),
                                        nn.BatchNorm2d(in_channels), nn.ReLU())
        self.conv_block = ResConBlock2d(in_channels, growth1=2, growth2=1)
        self.attention = attention
        if attention:
            self.attn_block = MultiviewAttentionBlock2d(out_channels, subband, head)

    def padding(self, length):

        length = math.ceil(length)
        length = math.ceil((length - self.kernel_size) / self.stride) + 1
        length = max(length, 1)
        length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        
        return int(length)
    
    def forward(self, x):
        """
            Input: [bn, feat, subband, time]
        """
        bn, feat, subband, time = x.size()

        padding = self.padding(subband)
        x = F.pad(x, (0, 0, 0 , padding - subband))
        x = self.down_conv(x)        
        x = self.conv_block(x)
        if self.attention:
            x = self.attn_block(x)

        return x

class Encoder2D_submain_causal(nn.Module):
  
    def __init__(self, in_channels, out_channels, kernel_size, stride, subband, head, attention=True):
        super().__init__()
        self.depth=1
        self.kernel_size, self.stride = kernel_size, stride
        self.down_conv  = nn.Sequential(nn.Conv2d(in_channels, in_channels, (kernel_size, 1), (stride, 1)),
                                        nn.BatchNorm2d(in_channels), nn.ReLU())
        self.conv_block = ResConBlock2d(in_channels, kernel_size=15, growth1=2, growth2=1)
        self.attention = attention
        if attention:
            self.attn_block = MultiviewAttentionBlock2d(out_channels, subband, head)

    def padding(self, length):

        length = math.ceil(length)
        length = math.ceil((length - self.kernel_size) / self.stride) + 1
        length = max(length, 1)
        length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        
        return int(length)
    
    def forward(self, x):
        """
            Input: [bn, feat, subband, time]
        """
        bn, feat, subband, time = x.size()

        padding = self.padding(subband)
        x = F.pad(x, (0, 0, 0 , padding - subband), mode="reflect")
        x = self.down_conv(x)        
        x = self.conv_block(x)
        if self.attention:
            x = self.attn_block(x)

        return x
        
class SubbandEncoder(nn.Module):
  
    def __init__(self, fullband_channel, subband_channel, kernel_size, stride, depth, head):
        super().__init__()
        encoders = []
        self.next_subband = subband_channel
        self.kernel_size, self.stride = kernel_size, stride
        for i in range(depth):
            self.next_subband = ((self.padding(self.next_subband) - (kernel_size-1) -1) // stride) + 1
            print(self.next_subband)
            encoders.append(
                Encoder2D(fullband_channel, fullband_channel, kernel_size, stride, self.next_subband, head)
            )
        
        self.encoders = nn.ModuleList(encoders)

    def padding(self, length):

        length = math.ceil(length)
        length = math.ceil((length - self.kernel_size) / self.stride) + 1
        length = max(length, 1)
        length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        
        return int(length)

    def forward(self, x):
        """
            Input: [bn, feat, subband, time]
        """
        for enc in self.encoders:
            x = enc(x)

        return x

class SubbandEncoder_submain(nn.Module):
  
    def __init__(self, fullband_channel, subband_channel, kernel_size, stride, depth, head):
        super().__init__()
        encoders = []
        self.next_subband = subband_channel
        
        for i in range(depth):
            self.next_subband = ((self.padding(self.next_subband, kernel_size, stride) - (kernel_size-1) -1) // stride) + 1
            print(self.next_subband)
            enc_use_att = True if i == depth-1 else False
            encoders.append(
                Encoder2D_submain(fullband_channel, fullband_channel, kernel_size, stride, self.next_subband, head, enc_use_att)
            )
            # kernel_size, stride = kernel_size // 4,  stride// 4
            kernel_size, stride = kernel_size // 4,  stride// 4
        
        self.encoders = nn.ModuleList(encoders)

    def padding(self, length, kernel_size, stride):

        length = math.ceil(length)
        length = math.ceil((length - kernel_size) / stride) + 1
        length = max(length, 1)
        length = (length - 1) * stride + kernel_size
        length = int(math.ceil(length))
        
        return int(length)

    def forward(self, x):
        """
            Input: [bn, feat, subband, time]
        """
        for enc in self.encoders:
            x = enc(x)

        return x

class Encoder2D_Causal(nn.Module):
  
    def __init__(self, in_channels, out_channels, kernel_size, stride, subband, head, attention=True, activation_fn='relu'):
        super().__init__()
        self.depth=1
        self.kernel_size, self.stride = kernel_size, stride
        if activation_fn == 'tanh':
            self.activation_fn = nn.Tanh
        else:
            self.activation_fn = nn.ReLU
        self.down_conv  = nn.Sequential(nn.Conv2d(in_channels, in_channels, (kernel_size, 1), (stride, 1)),
                                        nn.BatchNorm2d(in_channels), self.activation_fn())
        self.conv_block = ResConBlock2d_Causal(in_channels, kernel_size=15, growth1=4, growth2=1)
        self.attention = attention
        if attention:
            self.attn_block = MultiviewAttentionBlock2d(out_channels, subband, head)

    def padding(self, length):

        length = math.ceil(length)
        length = math.ceil((length - self.kernel_size) / self.stride) + 1
        length = max(length, 1)
        length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        
        return int(length)
    
    def forward(self, x):
        """
            Input: [bn, feat, subband, time]
        """
        bn, feat, subband, time = x.size()

        padding = self.padding(subband)
        x = F.pad(x, (0, 0, 0 , padding - subband), mode="reflect")
        x = self.down_conv(x)        
        x = self.conv_block(x)
        if self.attention:
            x = self.attn_block(x)

        return x

class SubbandEncoder_FullCausal(nn.Module):
  
    def __init__(self, fullband_channel, subband_channel, kernel_size, stride, depth, head):
        super().__init__()
        encoders = []
        self.next_subband = subband_channel
        
        for i in range(depth):
            self.next_subband = ((self.padding(self.next_subband, kernel_size, stride) - (kernel_size-1) -1) // stride) + 1
            print(self.next_subband)
            enc_use_att = True if i == depth-1 else False
            encoders.append(
                Encoder2D_Causal(fullband_channel, fullband_channel, kernel_size, stride, self.next_subband, head, enc_use_att)
            )
            kernel_size, stride = kernel_size // 4,  stride// 4
        
        self.encoders = nn.ModuleList(encoders)

    def padding(self, length, kernel_size, stride):

        length = math.ceil(length)
        length = math.ceil((length - kernel_size) / stride) + 1
        length = max(length, 1)
        length = (length - 1) * stride + kernel_size
        length = int(math.ceil(length))
        
        return int(length)

    def forward(self, x):
        """
            Input: [bn, feat, subband, time]
        """
        for enc in self.encoders:
            x = enc(x)

        return x

class SubbandEncoder_FullCausal_bottomup(nn.Module):
  
    def __init__(self, fullband_channel, subband_channel, kernel_size, stride, depth, head, activation_fn='relu'):
        super().__init__()
        encoders = []
        self.next_subband = subband_channel
        kernel_size=8
        stride=4
        
        for i in range(depth):
            self.next_subband = ((self.padding(self.next_subband, kernel_size, stride) - (kernel_size-1) -1) // stride) + 1
            print(self.next_subband)
            # enc_use_att = True if i == depth-1 else False
            enc_use_att = True #if i == 0 else False
            encoders.append(
                Encoder2D_Causal(fullband_channel, fullband_channel, kernel_size, stride, self.next_subband, head, enc_use_att, activation_fn=activation_fn)
            )
            kernel_size, stride = kernel_size // 2 ,  stride // 2
        
        self.encoders = nn.ModuleList(encoders)
        # print(self.encoders)

    def padding(self, length, kernel_size, stride):

        length = math.ceil(length)
        length = math.ceil((length - kernel_size) / stride) + 1
        length = max(length, 1)
        length = (length - 1) * stride + kernel_size
        length = int(math.ceil(length))
        
        return int(length)

    def forward(self, x):
        """
            Input: [bn, feat, subband, time]
        """
        for enc in self.encoders:
            x = enc(x)

        return x


if __name__ == '__main__':
    pass
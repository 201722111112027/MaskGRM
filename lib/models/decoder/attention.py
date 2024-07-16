# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : attention.py
# Copyright (c) Skye-Song. All Rights Reserved
import torch
import torch.nn as nn
from einops import rearrange


def mask_drop_out_dec(attn_weight, P):
    '''
    attn_weight: N, L, L; applied softmax
    mask: 0 is keep, 1 is masked
    '''
    prob_w = torch.zeros_like(attn_weight)
    iden_w = torch.ones_like(attn_weight)
    N, L, L = attn_weight.shape
    len_f = int(L//2)

    # sample queries; based on the softmax attention matrix
    attn_weight_softmax = attn_weight.softmax(dim=-1)
    # here w do not use the softmax for nomalization for better temporal matching
    pre_att_query = attn_weight_softmax[:, 0:len_f, len_f:]
    lat_att_query = attn_weight_softmax[:, len_f:, 0:len_f]
    att_query = torch.cat((pre_att_query, lat_att_query), dim=1) #bs, L, L//2
    # for max
    att_query_max = att_query.max(dim=-1).values + 1e-6 # bs, L

    # sample specific dropout positions for each query
    sample_num = int(P * (int(L//2)-1)) * L
    pre_att = attn_weight_softmax[:, 0:len_f, 0:len_f]  # bs, L//2, L//2
    pre_att = pre_att + torch.relu(-pre_att.min(dim=-1).values).unsqueeze(-1) + 1e-6    # bs, L//2, L//2
    lat_att = attn_weight_softmax[:, len_f:, len_f:]
    lat_att = lat_att + torch.relu(-lat_att.min(dim=-1).values).unsqueeze(-1) + 1e-6    # bs, L//2, L//2
    for i in range(N):
        pre_att[i].fill_diagonal_(0)
        lat_att[i].fill_diagonal_(0)
    spatial_att_all = torch.cat((pre_att, lat_att), dim=1) # bs, L, L//2
    spatial_att_all = spatial_att_all / spatial_att_all.sum(dim=-1).unsqueeze(-1) # bs, L, L//2
    att_final = spatial_att_all * att_query_max.unsqueeze(-1) # bs, L, L//2
    prob_flag = torch.zeros_like(att_final).view(N, -1)
    sam_indices = torch.multinomial(att_final.view(N, -1), sample_num)
    prob_flag.scatter_(dim=1, index=sam_indices, value=1)
    prob_flag = prob_flag.view(N, L, len_f)
    prob_w[:, 0:len_f, 0:len_f] = prob_flag[:, 0:len_f, 0:len_f]
    prob_w[:, len_f:, len_f:]   = prob_flag[:, len_f:, 0:len_f]

    return (prob_w == 1) * (iden_w == 1) # 1 is dropout



class Attention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
				 P=0.1, masked_only=None, P_query=None, mean_head=None):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

		self.P = P
		self.masked_only = masked_only
		self.P_query = P_query
		self.mean_head = mean_head

	def forward(self, x):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C//head)

		attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, head, N, N)

		bool_w = mask_drop_out_dec(attn.flatten(0, 1).detach().clone(), self.P)
		attn[bool_w.view(-1, self.num_heads, attn.shape[-2], attn.shape[-1])] = -float('inf')

		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x

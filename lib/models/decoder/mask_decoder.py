# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : vit_decoder.py
# Copyright (c) Skye-Song. All Rights Reserved

import torch
import torch.nn as nn
from einops import rearrange

from lib.utils.box_ops import box_xywh_to_cxywh, box_cxcywh_to_xyxy
from lib.models.decoder.block import Block
from lib.models.decoder.pos_embed import get_2d_sincos_pos_embed
# from external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from lib.utils.image import *

class MaskDecoder(nn.Module):
	def __init__(self, mask_ratio=0.75, patch_size=16, num_patches_z=64, num_patches_x=256,
				 embed_dim=1024, decoder_embed_dim=512,
	             decoder_depth=8, decoder_num_heads=16, pool_size=8,
	             mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
				 P=0.1, masked_only=False, d_step=0, P_query=None, mean_head=False):
		super().__init__()
		self.mask_ratio = mask_ratio

		self.num_patches = num_patches_z + num_patches_x
		self.num_patches_z = num_patches_z
		self.num_patches_x = num_patches_x
		self.patch_size = patch_size

		# self.search_prroipool = PrRoIPool2D(pool_size, pool_size, spatial_scale=1.0)

		self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

		self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

		self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_x, decoder_embed_dim),
		                                      requires_grad=False)  # fixed sin-cos embedding
		self.z_decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_z, decoder_embed_dim),
												requires_grad=False)

		self.decoder_blocks = nn.ModuleList([
			Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
				  P=P, masked_only=masked_only, P_query=P_query, mean_head=mean_head)
			for i in range(decoder_depth)])

		self.decoder_norm = norm_layer(decoder_embed_dim)
		self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3, bias=True)  # decoder to patch

		self.norm_pix_loss = norm_pix_loss

		self.initialize_weights()

	def initialize_weights(self):
		# initialize (and freeze) pos_embed by sin-cos embedding
		decoder_pos_embed_z = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
		                                            int(self.num_patches_z ** .5), cls_token=False)
		self.z_decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed_z).float().unsqueeze(0))

		decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
													int(self.num_patches_x ** .5), cls_token=False)
		self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

		# timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
		torch.nn.init.normal_(self.mask_token, std=.02)
		# initialize nn.Linear and nn.LayerNorm
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			# we use xavier_uniform following official JAX ViT:
			torch.nn.init.xavier_uniform_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	def random_masking(self, x):
		"""
		Perform per-sample random masking by per-sample shuffling.
		Per-sample shuffling is done by argsort random noise.
		x: [N, L, D], sequence
		"""
		N, L, D = x.shape  # batch, length, dim
		len_keep = int(L * (1 - self.mask_ratio))

		noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

		# sort noise for each sample
		ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
		ids_restore = torch.argsort(ids_shuffle, dim=1)

		# keep the first subset
		ids_keep = ids_shuffle[:, :len_keep]
		x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

		# generate the binary mask: 0 is keep, 1 is remove
		mask = torch.ones([N, L], device=x.device)
		mask[:, :len_keep] = 0
		# unshuffle to get the binary mask
		mask = torch.gather(mask, dim=1, index=ids_restore)

		# get the masked x
		mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x_keep.shape[1], 1)
		x_ = torch.cat([x_keep, mask_tokens], dim=1)  # no cls token
		x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

		return x_masked, mask

	def forward_decoder(self, z_feat, x_feat):
		# embed tokens
		z_feat = self.decoder_embed(z_feat)
		x_feat = self.decoder_embed(x_feat)

		# append mask tokens to sequence
		z_feat, mask_z = self.random_masking(z_feat)
		x_feat, mask_x = self.random_masking(x_feat)

		z_feat = z_feat + self.z_decoder_pos_embed
		x_feat = x_feat + self.decoder_pos_embed

		x = torch.cat([z_feat, x_feat], dim=1)
		mask = torch.cat([mask_z, mask_x], dim=1)

		# apply Transformer blocks
		for blk in self.decoder_blocks:
			x = blk(x)
		x = self.decoder_norm(x)

		# predictor projection
		x = self.decoder_pred(x)
		return x, mask

	def patchify(self, imgs):
		"""
		imgs: (N, 3, H, W)
		x: (N, L, patch_size**2 *3)
		"""
		p = self.patch_size
		assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

		h = w = imgs.shape[2] // p
		x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
		x = torch.einsum('nchpwq->nhwpqc', x)
		x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

		return x

	def forward_loss(self, imgs_x, imgs_z, pred, mask):
		"""
		imgs: [N, 3, H, W]
		pred: [N, L, p*p*3]
		mask: [N, L], 0 is keep, 1 is remove,
		"""
		target_x = self.patchify(imgs_x)
		target_z = self.patchify(imgs_z)
		target = torch.cat((target_z, target_x), dim=1)

		if self.norm_pix_loss:
			mean = target.mean(dim=-1, keepdim=True)
			var = target.var(dim=-1, keepdim=True)
			target = (target - mean) / (var + 1.e-6) ** .5

		loss = (pred - target) ** 2
		loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

		loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
		return loss


	# def crop_search_feat(self, search_feat, gt_bboxes):
	#
	# 	crop_bboxes = box_xywh_to_cxywh(gt_bboxes)
	# 	crop_sz = torch.sqrt(gt_bboxes[:, 2] * gt_bboxes[:, 3]) * 2.0
	# 	crop_sz = torch.clamp(crop_sz, min=0., max=1.)
	# 	crop_bboxes[:, 2] = crop_bboxes[:, 3] = crop_sz
	#
	# 	crop_bboxes = crop_bboxes * search_feat.shape[-1]
	# 	crop_bboxes = box_cxcywh_to_xyxy(crop_bboxes.clone().view(-1, 4))
	# 	batch_size = crop_bboxes.shape[0]
	# 	batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(crop_bboxes.device)
	#
	# 	target_roi = torch.cat((batch_index, crop_bboxes), dim=1)
	# 	search_box_feat = self.search_prroipool(search_feat, target_roi)
	# 	return search_box_feat



	def forward(self, x, imgs_z, imgs_x):
		z_feat, x_feat = torch.split(x, [self.num_patches_z, self.num_patches_x], dim=1)
		#
		# loss = torch.tensor(0.)
		# if gt_bboxes is not None:
		# 	x = rearrange(x_feat, 'b (h w) c -> b c h w', h=16, w=16).contiguous()
		# 	x = self.crop_search_feat(x, gt_bboxes)
		# 	x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
		# 	pred, mask = self.forward_decoder(x)  # [N, L, p*p*3]
		# 	loss = self.forward_loss(imgs=imgs_z, pred=pred, mask=mask)
		#
		# pred_z, mask_z = self.forward_decoder(z_feat)  # [N, L, p*p*3]
		# pred_x, mask_x = self.forward_decoder(x_feat)  # [N, L, p*p*3]
		#
		# loss_z = self.forward_loss(imgs=imgs_z, pred=pred_z, mask=mask_z)
		# loss_x = self.forward_loss(imgs=imgs_x, pred=pred_x, mask=mask_x)

		pred, mask = self.forward_decoder(z_feat, x_feat)
		loss = self.forward_loss(imgs_x, imgs_z, pred, mask)

		return loss


def mask_decoder():
	model = MaskDecoder(
		mask_ratio=0.75, patch_size=16, num_patches=8 ** 2, embed_dim=1024, decoder_embed_dim=512, decoder_depth=8,
		decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)
	return model


def build_maskdecoder(cfg):
	pool_size = int(cfg.DATA.TEMPLATE.SIZE / cfg.MODEL.BACKBONE.PATCHSIZE)

	num_patches_z = (cfg.DATA.TEMPLATE.SIZE // cfg.MODEL.BACKBONE.PATCHSIZE) ** 2
	num_patches_x = (cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.PATCHSIZE) ** 2

	model = MaskDecoder(
		mask_ratio=cfg.MODEL.DECODER.MASK_RATIO,
		patch_size=cfg.MODEL.BACKBONE.PATCHSIZE,
		num_patches_z=num_patches_z,
		num_patches_x=num_patches_x,
		embed_dim=cfg.MODEL.BACKBONE.EMBEDDIM,
		decoder_embed_dim=cfg.MODEL.DECODER.EMBEDDIM,
		decoder_depth=cfg.MODEL.DECODER.DEPTH,
		decoder_num_heads=cfg.MODEL.DECODER.NUMHEADS,
		pool_size=pool_size,
		mlp_ratio=cfg.MODEL.DECODER.MLPRATIO,
		norm_layer=nn.LayerNorm,
		norm_pix_loss=False)
	return model

"""
Basic GRM model.
"""

import os

import torch
from torch import nn

from lib.models.layers.head import build_box_head
from lib.models.grm.vit import vit_base_patch16_224_base, vit_base_patch16_224_large
from lib.utils.box_ops import box_xyxy_to_cxcywh


class GRM(nn.Module):
    """
    This is the base class for GRM.
    """

    def __init__(self, transformer, box_head, head_type='CORNER'):
        """
        Initializes the model.

        Parameters:
            transformer: Torch module of the transformer architecture.
        """

        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.head_type = head_type

    def forward(self, template, search, t_mask=None, s_mask=None, softmax=True, threshold=0.):
        x, _ = self.backbone(z=template, x=search, t_mask=t_mask, s_mask=s_mask, threshold=threshold)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        return out

    def forward_head(self, cat_feature, softmax=None, gt_score_map=None):
        """
        cat_feature: Output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C).
        """

        enc_opt = cat_feature[:, -256:]  # Encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, 16, 16)

        if self.head_type == 'CORNER_UP':
            # Run the corner head
            pred_boxes, prob_vec_tl, prob_vec_br = self.box_head(opt_feat, return_dist=True, softmax=softmax)
            outputs_coord = box_xyxy_to_cxcywh(pred_boxes)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                    'prob_tl': prob_vec_tl,
                    'prob_br': prob_vec_br}
            return out
        elif self.head_type == 'CENTER':
            # Run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_grm(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('GRM' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        # pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base':
        if cfg.MODEL.PRETRAIN_FILE == 'mae_pretrain_vit_base.pth':
            backbone = vit_base_patch16_224_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        elif cfg.MODEL.PRETRAIN_FILE == 'deit_base_patch16_224-b5f2ef4d.pth':
            backbone = vit_base_patch16_224_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        elif cfg.MODEL.PRETRAIN_FILE == 'deit_base_distilled_patch16_224-df68dfff.pth':
            backbone = vit_base_patch16_224_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, distilled=True)
            hidden_dim = backbone.embed_dim
            patch_start_index = 2
        else:
            raise NotImplementedError
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large':
        if cfg.MODEL.PRETRAIN_FILE == 'mae_pretrain_vit_large.pth':
            backbone = vit_base_patch16_224_large(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = GRM(
        backbone,
        box_head,
        head_type=cfg.MODEL.HEAD.TYPE
    )

    if 'GRM' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['net'], strict=False)
        print('load pretrained model from ' + cfg.MODEL.PRETRAIN_FILE)
    return model

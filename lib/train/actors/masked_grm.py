import torch
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from . import BaseActor
from ...utils.heapmap_utils import generate_heatmap
from ...utils.mask_utils import generate_mask_cond


class MaskedGRMActor(BaseActor):
    """
    Actor for training GRM models.
    """
    def __init__(self, net, objective, loss_weight, settings):
        super(MaskedGRMActor, self).__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # Batch size

    def __call__(self, data):
        """
        Args:
            data: The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)

        Returns:
            loss: The training loss.
            status: Dict containing detailed losses.
        """

        # Forward pass
        out_dict = self.forward_pass(data)
        gt_bboxes = data['search_anno']
        # Compute losses
        # loss, status = self.compute_losses(out_dict, gt_bboxes[0])
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        # Currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_img = list()
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_img.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)

        if len(template_img) == 1:
            template_img = template_img[0]

        gt_bboxes = data['search_anno'][0]
        out_dict = self.net(template=template_img, search=search_img, gt_bboxes=None)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True, labels=None):
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
                                            self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError('ERROR: network outputs is NAN! stop training')
        num_queries = pred_boxes.size(1)
        # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        # (B,4) --> (B,1,4) --> (B,N,4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)
        # Compute GIoU and IoU
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # Compute L1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            'focal'] * location_loss

        if 'renew_loss' in pred_dict:
            renew_loss = pred_dict['renew_loss']
            loss += self.loss_weight['renew'] * renew_loss

        if return_status:
            # Status for log
            mean_iou = iou.detach().mean()
            if 'renew_loss' in pred_dict:
                status = {'Ls/total': loss.item(),
                          'Ls/giou': giou_loss.item(),
                          'Ls/l1': l1_loss.item(),
                          'Ls/loc': location_loss.item(),
                          'Ls/renew_loss': renew_loss.item(),
                          'IoU': mean_iou.item()}
            else:
                status = {'Ls/total': loss.item(),
                          'Ls/giou': giou_loss.item(),
                          'Ls/loc': location_loss.item(),
                          'Ls/l1': l1_loss.item(),
                          'IoU': mean_iou.item()}
            return loss, status
        else:
            return loss

import os


class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/root/MaskGRM'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/root/MaskGRM/tensorboard'  # Directory for tensorboard files.
        self.pretrained_networks = '/root/MaskGRM/pretrained_networks'
        self.lasot_dir = '/hy-public/LaSOT'
        self.got10k_dir = '/hy-public/GOT10K/train'
        self.trackingnet_dir = '/hy-public/TrackingNet'
        self.coco_dir = '/hy-public/COCO'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''

import os

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here

    settings.davis_dir = ''
    settings.got10k_path = '/home/host/mounted2/TrackingDataset/GOT-10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = '/home/host/mounted2/TrackingDataset/LaSOT'
    settings.lasot_extension_subset_path = '/home/host/mounted2/TrackingDataset/LaSOT_Ext'
    settings.network_path = '/home/host/mounted2/gxm/third-work/MaskedGRM/networks'  # Where tracking networks are stored.
    settings.nfs_path = '/home/host/mounted2/TrackingDataset/track_test_dataset/nfs'
    settings.avist_path = '/home/host/mounted2/TrackingDataset/track_test_dataset/avist'
    settings.otb_path = '/home/host/mounted2/TrackingDataset/track_test_dataset/OTB100'
    settings.prj_dir = '/home/host/mounted2/gxm/third-work/MaskedGRM'
    settings.result_plot_path = '/home/host/mounted2/gxm/third-work/MaskedGRM/test/result_plots'
    settings.results_path = '/home/host/mounted2/gxm/third-work/MaskedGRM/test/tracking_results'  # Where to store tracking results
    settings.save_dir = '/home/host/mounted2/gxm/third-work/MaskedGRM'
    settings.segmentation_path = '/home/host/mounted2/gxm/third-work/MaskedGRM/test/segmentation_results'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/host/mounted2/TrackingDataset/TrackingNet'
    settings.uav_path = '/home/host/mounted2/TrackingDataset/track_test_dataset/UAV123'
    settings.vot_path = '/home/host/mounted2/TrackingDataset/track_test_dataset/VOT2019'
    settings.youtubevos_dir = ''

    return settings

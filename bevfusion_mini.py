_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# --- 1. Define Paths ---
data_root = 'data/nuscenes/'
ann_file = 'nuscenes_infos_val.pkl'

# --- 2. Define the Test Dataloader (Mini) ---
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=dict(
            img='samples/',
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            sweeps='sweeps/LIDAR_TOP'),
        pipeline=[
            dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color', backend_args=None),
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=None),
            dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True, backend_args=None),
            dict(type='ImageAug3D', final_dim=[256, 704], resize_lim=[0.48, 0.48], bot_pct_lim=[0.0, 0.0], rot_lim=[0.0, 0.0], rand_flip=False, is_train=False),
            dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
            dict(
                type='Pack3DDetInputs',
                keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
                meta_keys=['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar', 'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx', 'lidar_path', 'img_path', 'num_pts_feats'])
        ],
        metainfo=dict(classes=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']),
        modality=dict(use_lidar=True, use_camera=True),
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=None
    )
)

# --- 3. THE FIX: Dummy Evaluator that saves results to a specific directory ---
test_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + ann_file,
    metric='bbox',
    format_only=True, # Do not calculate scores (skip the crashy part)
    jsonfile_prefix='save_output' # This prefix is used for the output JSON
)

# We must ensure val loops are disabled
val_dataloader = None
val_evaluator = None
val_cfg = None

#test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1), 
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=-1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # --- THE VISUALIZATION HOOK (REQUIRED FOR IMAGES) ---
    visualization=dict(
        type='Det3DVisualizationHook',
        draw=True,
        show=False,
        wait_time=0,
        test_out_dir='output_results_final',
        vis_task='multi-modality_det'
    )
)

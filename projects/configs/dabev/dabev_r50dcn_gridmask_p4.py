_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]

backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# -------------------------
# Space & data normalization
# -------------------------
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
test_classes = ['car', 'truck', 'bus', 'bicycle', 'pedestrian']
input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False)

# =========================
# 1) 模型（含 2D 解码 + 多标签头、BEV 分支、QAL、QST）
# =========================
model = dict(
    type='DAPetr3D',
    use_grid_mask=True,

    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),   # C4, C5
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=False,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        pretrained='ckpts/resnet50_msra-5891d200.pth',
    ),
    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2
    ),

    pts_bbox_head=dict(
        type='DAPETRHead',              # 代码里已扩展支持 DA/QAL/QST/BEV/2D-Aux
        num_classes=10,
        in_channels=256,
        num_query=600,

        # --- PETR 几何相关 ---
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,

        # --- 主 Transformer ---
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                        dict(type='PETRMultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=False,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                ),
            )
        ),
        positional_encoding=dict(type='SinePositionalEncoding3D', num_feats=128, normalize=True),

        # --- 编码/损失（3D主任务） ---
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),

        # --- 新增：BEV 分支（默认复用主 transformer） ---
        num_bev_query=600,
        bev_num_layers=6,
        use_separate_bev_decoder=False,

        # --- 新增：2D Image-View Decoder + 多标签分类头（论文图左侧绿框） ---
        img_view_decoder=dict(         # D^iv
            type='ImageViewDecoder',   # 在 head 内部解析并用标准 TransformerDecoder 实现
            num_layers=3,
            num_queries=300,
            embed_dims=256,
            num_heads=8,
            ffn_dim=1024,
            dropout=0.1,
            return_intermediate=False
        ),
        img_view_head=dict(            # H_cls
            type='MultilabelLinearHead',
            in_channels=256,
            num_classes=10,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)  # L^iv 权重
        ),

        # --- 新增：域自适应 / QAL ---
        da_cfg=dict(
            use_img_da=True,            # 图像特征域对齐
            use_query_da=True,          # BEV 查询域对齐
            use_aux_2d_head=True,       # 开启 2D 多标签辅助分支
            # 不依赖 hook 也能跑：常量 GRL；若想 warmup，把 init=0,max=1,warmup_epochs=5，并加 Hook
            grl=dict(init=0.0, max=0.2, warmup_epochs=3),
            loss_weights=dict(img_da=0.02, query_da=0.02, aux2d=1.0, bev_ent=0.01)  # aux2d 是 2D 头的附加权重
        ),

        # --- 新增：QST 伪标签（用于图中两条“自训练”虚线） ---
        qst_cfg=dict(
            enable=True,
            queue_len=50,     # 公式(11)
            ema_gamma=0.9,    # 公式(12)
            top_percent=0.2,  # 公式(13) 的 v
            loss_weight=0.5,  # L_qst 权重
            use_det_loss=False
        ),

        # --- 参考点/PE 稳健化 ---
        query_init_cfg=dict(pe_noise_std=0.01, freeze_pe_epochs=0),
    ),

    # --- 训练与测试设置 ---
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range
            )
        )
    ),
)

model['pts_bbox_head']['da_cfg']['grl'] = dict(init=0.0, max=0.5, warmup_epochs=3)
model['pts_bbox_head']['da_cfg']['loss_weights'] = dict(img_da=0.2, query_da=0.2, aux2d=0.1)
model['pts_bbox_head']['qst_cfg']['loss_weight'] = 0.5 # 原来 0.5

model['pts_bbox_head']['da_cfg']['disc_detach'] = False  # 默认 False，改 True 可微调
model['pts_bbox_head']['aux_dec_weight'] = 0.5
model['pts_bbox_head']['da_cfg'].update(img_disc=dict(in_dim=256), bev_disc=dict(in_dim=256))
model['pts_bbox_head']['da_cfg'].update(dict(use_query_da=True))
# model['pts_bbox_head']['da_cfg']['loss_weights'].update(dict(query_da=0.02))  # 先小一点

# =========================
# 2) 数据集（源/目标混采；目标域不加载 GT，并在 img_metas 写入 domain）
# =========================
dataset_type = 'DACustomNuScenesDataset'
dataset_type_N = 'CustomNuScenesDataset'
# dataset_type_L = 'LyftDataset'
src_root = '/data/dataset/nuScenes/'
tgt_root = '/data/dataset/lyft/'

file_client_args = dict(backend='disk')
db_sampler = dict()
ida_aug_conf = dict(
    resize_lim=(0.8, 1.0),
    final_dim=(512, 1408),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900, W=1600,
    rand_flip=True,
)

_meta_keys = (
    'filename', 'ori_shape', 'img_shape', 'pad_shape', 'img_norm_cfg',
    'lidar2img', 'intrinsics', 'extrinsics',
    'timestamp', 'img_timestamp',
    'scale_factor', 'flip', 'flip_direction',
    # ★ 关键：把 domain 放进去
    'domain',
    'box_type_3d', 'box_mode_3d', 
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925], translation_std=[0, 0, 0],
         scale_ratio_range=[0.95, 1.05], reverse_angle=True, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='AddDomainToMetas', domain='source'),   # 必须在 Collect3D 之前
    dict(type='Collect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
         meta_keys=_meta_keys),
]

tgt_train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='EnsureEmptyGT3D'),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925], translation_std=[0, 0, 0],
         scale_ratio_range=[0.95, 1.05], reverse_angle=True, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='AddDomainToMetas', domain='target'),   # 必须在 Collect3D 之前
    dict(type='Collect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
         meta_keys=_meta_keys),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
             dict(type='Collect3D', keys=['img'], meta_keys=_meta_keys),
         ])
]
# val=dict(type=dataset_type_N, 
#          data_root=src_root,
#          ann_file=src_root + 'nuScenes_infos_val.pkl', 
#          pipeline=test_pipeline, classes=class_names, modality=input_modality),
# test=dict(type=dataset_type_N,
#           data_root=src_root,
#           ann_file=src_root + 'nuScenes_infos_val.pkl', 
#           pipeline=test_pipeline, classes=class_names, modality=input_modality)
val=dict(type=dataset_type, 
         data_root=src_root,
         ann_file=src_root + 'nuScenes_infos_val.pkl', 
         pipeline=test_pipeline, classes=class_names, modality=input_modality,
         test_mode=True,
         box_type_3d='LiDAR'),
test=dict(type=dataset_type,
          data_root=tgt_root,
          ann_file=tgt_root + 'lyft_infos_ann_val.pkl', 
          pipeline=test_pipeline, classes=class_names, modality=input_modality,
          test_mode=True,
          box_type_3d='LiDAR')


data = dict(
    samples_per_gpu=3,
    workers_per_gpu=8,
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                domain='source',
                data_root=src_root,
                ann_file=src_root + 'nuScenes_infos_train.pkl',
                pipeline=train_pipeline,
                classes=class_names,
                modality=input_modality,
                test_mode=False,
                use_valid_flag=True,
                box_type_3d='LiDAR',
                use_ann=True),
            dict(
                type=dataset_type,
                domain='target',
                data_root=tgt_root,
                ann_file=tgt_root + 'lyft_infos_ann_val.pkl',
                pipeline=tgt_train_pipeline,
                classes=class_names,
                modality=input_modality,
                test_mode=False,
                use_valid_flag=True,
                box_type_3d='LiDAR',
                use_ann=False,
                filter_empty_gt=False)
        ]),
    # val=dict(type=dataset_type, pipeline=test_pipeline, classes=class_names, modality=input_modality),
    # test=dict(type=dataset_type, pipeline=test_pipeline, classes=class_names, modality=input_modality)
    val=val,
    test=test,
    persistent_workers=True,
    pin_memory=True
)

# =========================
# 3) 优化器 / 训练策略
# =========================
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    weight_decay=0.01
)

# 开启 fp16
fp16 = dict(loss_scale='dynamic')

optimizer_config = dict(
    grad_clip=dict(max_norm=3, norm_type=2),
    coalesce=True,
    bucket_size_mb=-1,
)


# 稳定 DA 初期
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

total_epochs = 24
evaluation = dict(interval=2, pipeline=test_pipeline)  # 更频繁看曲线
find_unused_parameters = True
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = None
resume_from = None


custom_hooks = [
    dict(type='FirstBatchProbeHook'),
    dict(type='GRLSchedulerHook'),   # 若你用了 grl warmup
    dict(type='DAQSTMonitorHook'),
    dict(type='ClipDAHook', max_norm_disc=0.5, priority='HIGH'),
]
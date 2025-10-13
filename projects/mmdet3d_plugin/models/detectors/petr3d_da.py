# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from os import path as osp
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core import (
    CameraInstance3DBoxes,
    LiDARInstance3DBoxes,
    bbox3d2result,
    show_multi_modality_result,
)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class DAPetr3D(MVXTwoStageDetector):
    """DAPetr3D detector.

    本版本仅做了一个“安全补丁”：
    - forward_train 允许 batch 中出现**无 GT 的样本**（目标域），自动补空实例，
      以配合 PETRHead 中的 DA/QAL/QST/2D-Aux 训练逻辑。
    """

    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(DAPetr3D, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=("img"), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): GT boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): GT labels for boxes of each sample.
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): GT boxes to be ignored.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        
        # ==== 新增：缓存（已 detach）====
        try:
            cached = {}
            for k, v in outs.items():
                cached[k] = v.detach() if torch.is_tensor(v) else v
            self.last_preds = cached
            self.last_metas = img_metas
        except Exception:
            pass
        # ==============================
        
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    @force_fp32(apply_to=("img", "points"))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether return_loss=True."""
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def _ensure_gt_lists(self, B, device, gt_bboxes_3d, gt_labels_3d):
        """把 None 或缺失的 GT 替换为“空实例”，确保下游 assigner/loss 不报错。"""
        if gt_bboxes_3d is None or gt_labels_3d is None:
            gt_bboxes_3d = [None] * B
            gt_labels_3d = [None] * B

        new_gt_bboxes_3d, new_gt_labels_3d = [], []
        for i in range(B):
            boxes_i = gt_bboxes_3d[i]
            labels_i = gt_labels_3d[i]

            if boxes_i is None:
                empty_boxes = LiDARInstance3DBoxes(
                    torch.zeros((0, 7), dtype=torch.float32, device=device),
                    box_dim=7,
                    with_yaw=True,
                )
                new_gt_bboxes_3d.append(empty_boxes)
            else:
                # 保持在当前设备上
                if hasattr(boxes_i, "to"):
                    boxes_i = boxes_i.to(device)
                new_gt_bboxes_3d.append(boxes_i)

            if labels_i is None:
                new_gt_labels_3d.append(torch.zeros((0,), dtype=torch.long, device=device))
            else:
                new_gt_labels_3d.append(labels_i.to(device))

        return new_gt_bboxes_3d, new_gt_labels_3d

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
    ):
        """Forward training function.

        关键改造：支持 batch 中混入“无 GT”的目标域样本。
        - 自动将缺失的 gt_* 替换为长度为 0 的实例，保证与原有训练流程一致。
        """
        assert isinstance(img_metas, list)
        # 兼容 DataContainer 情况
        # imgs: (B, N, C, H, W)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # 构造/补齐 GT（为目标域或 None 的样本补空）
        # 设备以图像张量为准
        device = img[0].device if isinstance(img, list) else img.device
        B = len(img_metas)
        gt_bboxes_3d, gt_labels_3d = self._ensure_gt_lists(B, device, gt_bboxes_3d, gt_labels_3d)

        # 正常前向与损失
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore
        )
        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got {type(var)}")
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return bbox_results

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return bbox_list

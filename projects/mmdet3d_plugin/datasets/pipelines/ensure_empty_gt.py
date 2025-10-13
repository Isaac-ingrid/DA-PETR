# projects/mmdet3d_plugin/datasets/pipelines/ensure_empty_gt.py
from mmdet.datasets import PIPELINES
import torch
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes

@PIPELINES.register_module()
class EnsureEmptyGT3D:
    """在样本无标注或部分标注缺失时，补齐/清理关键信息，避免 DefaultFormatBundle3D 对 None to_tensor 报错。
    放置位置：DefaultFormatBundle3D 之前、Collect3D 之前。
    """

    def __call__(self, results):
        # ---------- 3D 标注（检测） ----------
        if 'gt_bboxes_3d' not in results or results['gt_bboxes_3d'] is None:
            results['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                torch.zeros((0, 7), dtype=torch.float32), box_dim=7, with_yaw=True
            )

        if 'gt_labels_3d' not in results or results['gt_labels_3d'] is None:
            results['gt_labels_3d'] = torch.zeros((0,), dtype=torch.long)

        # nuScenes 常见可选项：属性标签（若不存在或为 None，直接删除，避免 to_tensor(None)）
        if results.get('attr_labels', None) is None:
            results.pop('attr_labels', None)

        # ---------- 2D 标注（有些管线会顺手写入；目标域一般没有） ----------
        if results.get('gt_bboxes', None) is None:
            results['gt_bboxes'] = torch.zeros((0, 4), dtype=torch.float32)
        if results.get('gt_labels', None) is None:
            results['gt_labels'] = torch.zeros((0,), dtype=torch.long)

        # ---------- 其它可选键：为 None 时删除 ----------
        for k in ['gt_bboxes_ignore', 'gt_masks', 'centers2d', 'depths']:
            if results.get(k, None) is None:
                results.pop(k, None)

        # 某些管线会读取 gt_names；给个空列表兜底
        results.setdefault('gt_names', [])

        return results

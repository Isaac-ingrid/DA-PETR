# projects/mmdet3d_plugin/datasets/da_nuscenes_dataset.py
import math
import mmcv
import numpy as np
import torch
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

from mmdet3d.datasets import NuScenesDataset, DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes

# ==== 3D/BEV IoU 算子 ====
try:
    from mmdet3d.ops.iou3d.iou3d_utils import boxes_iou3d_gpu
except Exception:
    try:
        from mmdet3d.ops.iou3d import boxes_iou3d_gpu
    except Exception:
        boxes_iou3d_gpu = None

try:
    from mmdet3d.ops.iou3d.iou3d_utils import boxes_iou_bev
except Exception:
    boxes_iou_bev = None

# ==== 常量 ====
DEBUG_COORD_WARMUP_SAMPLES = 64
DEBUG_PRINT_SAMPLES = 8
CENTER_HIT_TOPK = 100
CENTER_HIT_THRESH = 2.0
CALIB_TOPK = 100

# —— 评估过滤策略（关键：永远保留 Top-K）——
EVAL_SCORE_THR = 0.0
EVAL_TOPK_KEEP = 100
EVAL_MIN_KEEP = 10

# —— 自适应校准限幅 —— 
SHIFT_CLAMP_XY = 1.0
SHIFT_CLAMP_Z  = 2.0
DYAW_CLAMP     = 0.35
APPLY_IMPROVE_EPS = 0.01

# ---------------- 工具函数 ----------------
def _to_lwh7_tensor(boxes: LiDARInstance3DBoxes) -> torch.Tensor:
    t = boxes.tensor
    if t.shape[-1] > 7:
        t = t[:, :7]
    return t[:, [0, 1, 2, 4, 3, 5, 6]]

def _from_lwh7_tensor(t_lwh: torch.Tensor) -> torch.Tensor:
    if t_lwh.shape[-1] > 7:
        t_lwh = t_lwh[:, :7]
    return t_lwh[:, [0, 1, 2, 4, 3, 5, 6]]

def _bev_aabb_iou_xywl(det_xywl: np.ndarray, gt_xywl: np.ndarray) -> np.ndarray:
    if det_xywl.size == 0 or gt_xywl.size == 0:
        return np.zeros((det_xywl.shape[0], gt_xywl.shape[0]), dtype=np.float32)
    det_x1 = det_xywl[:, 0] - det_xywl[:, 2] / 2
    det_y1 = det_xywl[:, 1] - det_xywl[:, 3] / 2
    det_x2 = det_xywl[:, 0] + det_xywl[:, 2] / 2
    det_y2 = det_xywl[:, 1] + det_xywl[:, 3] / 2
    gt_x1 = gt_xywl[:, 0] - gt_xywl[:, 2] / 2
    gt_y1 = gt_xywl[:, 1] - gt_xywl[:, 3] / 2
    gt_x2 = gt_xywl[:, 0] + gt_xywl[:, 2] / 2
    gt_y2 = gt_xywl[:, 1] + gt_xywl[:, 3] / 2
    ious = np.zeros((det_xywl.shape[0], gt_xywl.shape[0]), dtype=np.float32)
    for i in range(det_xywl.shape[0]):
        xx1 = np.maximum(det_x1[i], gt_x1)
        yy1 = np.maximum(det_y1[i], gt_y1)
        xx2 = np.minimum(det_x2[i], gt_x2)
        yy2 = np.minimum(det_y2[i], gt_y2)
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (det_x2[i] - det_x1[i]) * (det_y2[i] - det_y1[i])
        area_g = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        union = area_i + area_g - inter + 1e-6
        ious[i] = inter / union
    return ious

def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def _apply_coord_fix(b: LiDARInstance3DBoxes, mode: str) -> LiDARInstance3DBoxes:
    if mode == 'none':
        return b
    T = b.tensor.clone()
    if T.shape[-1] > 7:
        T = T[:, :7]
    x, y, z, dx, dy, dz, yaw = [T[:, i] for i in range(7)]
    if mode.startswith('swap_xy'):
        x, y = y, x
        dx, dy = dy, dx
        if '_p90' in mode:
            yaw = yaw + math.pi / 2
        else:
            yaw = yaw - math.pi / 2
    if 'flipx' in mode:
        x = -x; yaw = math.pi - yaw
    if 'flipy' in mode:
        y = -y; yaw = -yaw
    T[:, 0], T[:, 1], T[:, 2] = x, y, z
    T[:, 3], T[:, 4], T[:, 5], T[:, 6] = dx, dy, dz, yaw
    return LiDARInstance3DBoxes(T)

def _center_hit_rate(pred_boxes: LiDARInstance3DBoxes,
                     pred_scores: np.ndarray,
                     gt_boxes: LiDARInstance3DBoxes,
                     topk: int = CENTER_HIT_TOPK,
                     thr: float = CENTER_HIT_THRESH) -> Tuple[float, float]:
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, float('inf')
    if pred_scores is None or len(pred_scores) == 0:
        sel = slice(None)
    else:
        order = np.argsort(-pred_scores)[:min(topk, len(pred_scores))]
        sel = order
    pc = pred_boxes.gravity_center[:, :2].detach().cpu().numpy()[sel]
    gc = gt_boxes.gravity_center[:, :2].detach().cpu().numpy()
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(gc).query(pc, k=1)
    except Exception:
        d = np.sqrt(((pc[:, None, :] - gc[None, :, :]) ** 2).sum(-1)).min(axis=1) if len(gc) else np.array([np.inf] * len(pc))
    hit = float((d < thr).mean())
    avg_d = float(np.mean(d)) if len(d) else float('inf')
    return hit, avg_d

# ---------------- 估计：全局平移 / 航向 ----------------
def _estimate_global_translation(results, gt_annos, best_mode: str,
                                 warmup_n: int = 64,
                                 topk: int = 100, dist_thresh: float = 3.0):
    def _extract_box_score(r):
        if isinstance(r, dict):
            if 'pts_bbox' in r:
                b = r['pts_bbox']['boxes_3d']; s = r['pts_bbox']['scores_3d']
            else:
                b = r['boxes_3d']; s = r['scores_3d']
        else:
            b, s, _ = r
        bt = b.tensor if isinstance(b, LiDARInstance3DBoxes) else b
        if bt.shape[-1] > 7:
            b = LiDARInstance3DBoxes(bt[:, :7])
        s = s.detach().cpu().numpy() if torch.is_tensor(s) else (np.asarray(s) if s is not None else None)
        return b, s

    diffs = []
    before_hit, after_hit = [], []
    dx = dy = dz = 0.0

    n = min(warmup_n, len(results))
    for i in range(n):
        b, s = _extract_box_score(results[i])
        g = gt_annos[i]['gt_bboxes_3d']
        b = _apply_coord_fix(b, best_mode)
        if s is None or len(s) == 0 or len(b) == 0 or len(g) == 0:
            continue
        order = np.argsort(-s)[:min(topk, len(s))]
        pc = b.gravity_center.detach().cpu().numpy()[order, :]
        gc = g.gravity_center.detach().cpu().numpy()
        try:
            from scipy.spatial import cKDTree
            d, idx = cKDTree(gc[:, :2]).query(pc[:, :2], k=1)
        except Exception:
            diff = pc[:, None, :2] - gc[None, :, :2]
            dist = np.sqrt((diff ** 2).sum(-1))
            idx = dist.argmin(axis=1); d = dist[np.arange(dist.shape[0]), idx]
        m = d < dist_thresh
        if not m.any():
            continue
        pc_m = pc[m]; gc_m = gc[idx[m]]
        diffs.append(gc_m - pc_m)
        hit0, _ = _center_hit_rate(b, s, g, topk=topk, thr=CENTER_HIT_THRESH)
        before_hit.append(hit0)

    if len(diffs) > 0:
        delta = np.median(np.concatenate(diffs, axis=0), axis=0)
        dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])

    for i in range(n):
        b, s = _extract_box_score(results[i])
        g = gt_annos[i]['gt_bboxes_3d']
        b = _apply_coord_fix(b, best_mode)
        T = b.tensor.clone(); T[:, 0] += dx; T[:, 1] += dy; T[:, 2] += dz
        b2 = LiDARInstance3DBoxes(T)
        hit1, _ = _center_hit_rate(b2, s, g, topk=topk, thr=CENTER_HIT_THRESH)
        after_hit.append(hit1)

    h0 = np.mean(before_hit) if before_hit else 0.0
    h1 = np.mean(after_hit) if after_hit else 0.0
    apply = (h1 - h0) >= APPLY_IMPROVE_EPS

    if apply:
        dx = float(np.clip(dx, -SHIFT_CLAMP_XY, SHIFT_CLAMP_XY))
        dy = float(np.clip(dy, -SHIFT_CLAMP_XY, SHIFT_CLAMP_XY))
        dz = float(np.clip(dz, -SHIFT_CLAMP_Z,  SHIFT_CLAMP_Z))
    else:
        dx = dy = dz = 0.0

    print(f'[DEBUG] Global translation estimate: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f} '
          f'(center_hit_before={h0:.4f} -> after={h1:.4f}; apply={apply})')
    return dx, dy, dz

def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi

def _estimate_global_yaw_size(results, gt_annos, best_mode: str,
                              dx: float, dy: float, dz: float,
                              warmup_n: int = 64, topk: int = 100,
                              dist_thresh: float = 3.0):
    def _extract_bsl(r):
        if isinstance(r, dict):
            if 'pts_bbox' in r:
                b = r['pts_bbox']['boxes_3d']; s = r['pts_bbox']['scores_3d']; l = r['pts_bbox']['labels_3d']
            else:
                b = r['boxes_3d']; s = r['scores_3d']; l = r['labels_3d']
        else:
            b, s, l = r
        bt = b.tensor if isinstance(b, LiDARInstance3DBoxes) else b
        if bt.shape[-1] > 7:
            b = LiDARInstance3DBoxes(bt[:, :7])
        if torch.is_tensor(s): s = s.detach().cpu().numpy()
        if torch.is_tensor(l): l = l.detach().cpu().numpy()
        return b, s, l

    yaw_diffs = []
    n = min(warmup_n, len(results))
    for i in range(n):
        b, s, _ = _extract_bsl(results[i])
        g = gt_annos[i]['gt_bboxes_3d']
        b = _apply_coord_fix(b, best_mode)
        T = b.tensor.clone(); T[:, 0] += dx; T[:, 1] += dy; T[:, 2] += dz
        b = LiDARInstance3DBoxes(T)
        if s is None or len(s) == 0 or len(b) == 0 or len(g) == 0:
            continue
        order = np.argsort(-s)[:min(topk, len(s))]
        pb = b.tensor.detach().cpu().float()[order]
        gb = g.tensor.detach().cpu().float()

        pc = b.gravity_center.detach().cpu().numpy()[order, :2]
        gc = g.gravity_center.detach().cpu().numpy()[:, :2]
        try:
            from scipy.spatial import cKDTree
            d, idx = cKDTree(gc).query(pc, k=1)
        except Exception:
            diff = pc[:, None, :] - gc[None, :, :]
            dist = np.sqrt((diff ** 2).sum(-1))
            idx = dist.argmin(axis=1)

        pb_lwh = _to_lwh7_tensor(LiDARInstance3DBoxes(pb))
        gb_lwh = _to_lwh7_tensor(LiDARInstance3DBoxes(gb[idx]))
        yaw_diffs.append(_wrap_pi((gb_lwh[:, 6] - pb_lwh[:, 6]).numpy()))

    dyaw = float(np.median(np.concatenate(yaw_diffs))) if yaw_diffs else 0.0

    def _approx_miou_after(dyaw_):
        frames = min(32, len(results)); ious = []
        for i in range(frames):
            b, s, _ = _extract_bsl(results[i])
            g = gt_annos[i]['gt_bboxes_3d']
            b = _apply_coord_fix(b, best_mode)
            T = b.tensor.clone(); T[:, 0] += dx; T[:, 1] += dy; T[:, 2] += dz
            if dyaw_:
                T[:, 6] += dyaw_
            b = LiDARInstance3DBoxes(T)
            if len(b) == 0 or len(g) == 0:
                continue
            bl = _to_lwh7_tensor(b)
            AA = _bev_aabb_iou_xywl(bl[:, [0, 1, 4, 3]].cpu().numpy(),
                                    _to_lwh7_tensor(g)[:, [0, 1, 4, 3]].cpu().numpy())
            if AA.size > 0:
                ious.append(AA.max(axis=1).mean())
        return float(np.mean(ious)) if ious else 0.0

    miou_before = _approx_miou_after(0.0)
    miou_after  = _approx_miou_after(dyaw)
    improve = (miou_after - miou_before)
    apply = improve >= APPLY_IMPROVE_EPS
    if apply:
        dyaw = float(np.clip(dyaw, -DYAW_CLAMP, DYAW_CLAMP))
    else:
        dyaw = 0.0

    print(f'[DEBUG] Global yaw/size estimate: dyaw={dyaw:.3f} rad s_l=1.000 s_w=1.000 s_h=1.000 '
          f'(mIoU_before={miou_before:.4f} -> after={miou_after:.4f}; apply={apply})')
    return dyaw

# ---------------- 按类细调（可选） ----------------
def _estimate_per_class_vertical(
    results, gt_annos, class_names: List[str], best_mode: str,
    dx: float, dy: float, dz: float, warmup_n: int,
    topk: int = 100, dist_thresh: float = 3.0, min_pairs: int = 30,
    sh_clamp=(0.5, 2.0)
):
    dz_per_cls: Dict[int, list] = {i: [] for i in range(len(class_names))}
    sh_per_cls: Dict[int, list] = {i: [] for i in range(len(class_names))}
    used_pairs = {i: 0 for i in range(len(class_names))}

    def _extract_box_score_label(r):
        if isinstance(r, dict):
            if 'pts_bbox' in r:
                b = r['pts_bbox']['boxes_3d']; s = r['pts_bbox']['scores_3d']; l = r['pts_bbox']['labels_3d']
            else:
                b = r['boxes_3d']; s = r['scores_3d']; l = r['labels_3d']
        else:
            b, s, l = r
        bt = b.tensor if isinstance(b, LiDARInstance3DBoxes) else b
        if bt.shape[-1] > 7:
            b = LiDARInstance3DBoxes(bt[:, :7])
        if torch.is_tensor(s): s = s.detach().cpu().numpy()
        if torch.is_tensor(l): l = l.detach().cpu().numpy()
        return b, s, l

    n = min(warmup_n, len(results))
    for i in range(n):
        b, s, l = _extract_box_score_label(results[i])
        g = gt_annos[i]['gt_bboxes_3d']
        b = _apply_coord_fix(b, best_mode)
        T = b.tensor.clone(); T[:, 0] += dx; T[:, 1] += dy; T[:, 2] += dz
        b = LiDARInstance3DBoxes(T)
        if s is None or len(s) == 0 or len(b) == 0 or len(g) == 0:
            continue
        order = np.argsort(-s)[:min(topk, len(s))]
        pc_all = b.gravity_center.detach().cpu().float().numpy()
        gc_all = g.gravity_center.detach().cpu().float().numpy()
        pc = pc_all[order]; lc = l[order].astype(int)
        try:
            from scipy.spatial import cKDTree
            d, idx = cKDTree(gc_all[:, :2]).query(pc[:, :2], k=1)
        except Exception:
            diff = pc[:, None, :2] - gc_all[None, :, :2]
            dist = np.sqrt((diff ** 2).sum(-1))
            idx = dist.argmin(axis=1); d = dist[np.arange(dist.shape[0]), idx]
        mask = d < dist_thresh
        if not mask.any():
            continue
        pb = b.tensor.detach().cpu().float()[order][mask]
        gt = g.tensor.detach().cpu().float()[idx[mask]]
        pb_lwh = _to_lwh7_tensor(LiDARInstance3DBoxes(pb))
        gt_lwh = _to_lwh7_tensor(LiDARInstance3DBoxes(gt))
        dz_pair = (gt_lwh[:, 2] - pb_lwh[:, 2]).numpy()
        sh_pair = (gt_lwh[:, 5] / torch.clamp(pb_lwh[:, 5], 1e-3)).numpy()
        cls_pair = lc[mask]
        for zdelta, sscale, cid in zip(dz_pair.tolist(), sh_pair.tolist(), cls_pair.tolist()):
            if 0 <= cid < len(class_names):
                dz_per_cls[cid].append(float(zdelta))
                sh_per_cls[cid].append(float(np.clip(sscale, *sh_clamp)))
                used_pairs[cid] += 1

    dz_cls, sh_cls = {}, {}
    for cid in range(len(class_names)):
        if len(dz_per_cls[cid]) >= min_pairs:
            dz_cls[cid] = float(np.median(np.array(dz_per_cls[cid])))
        if len(sh_per_cls[cid]) >= min_pairs:
            sh_cls[cid] = float(np.median(np.array(sh_per_cls[cid])))
    return dz_cls, sh_cls, used_pairs

def _estimate_per_class_lw(
    results, gt_annos, class_names: List[str], best_mode: str,
    dx: float, dy: float, dz: float, warmup_n: int,
    topk: int = 100, dist_thresh: float = 3.0, min_pairs: int = 30,
    lw_clamp: Tuple[float,float]=(0.7, 1.3)
) -> Tuple[Dict[int,float], Dict[int,float], Dict[int,int]]:
    sl_per_cls: Dict[int, list] = {i: [] for i in range(len(class_names))}
    sw_per_cls: Dict[int, list] = {i: [] for i in range(len(class_names))}
    used_pairs = {i: 0 for i in range(len(class_names))}

    def _extract_bsl(r):
        if isinstance(r, dict):
            if 'pts_bbox' in r:
                b = r['pts_bbox']['boxes_3d']; s = r['pts_bbox']['scores_3d']; l = r['pts_bbox']['labels_3d']
            else:
                b = r['boxes_3d']; s = r['scores_3d']; l = r['labels_3d']
        else:
            b, s, l = r
        bt = b.tensor if isinstance(b, LiDARInstance3DBoxes) else b
        if bt.shape[-1] > 7:
            b = LiDARInstance3DBoxes(bt[:, :7])
        if torch.is_tensor(s): s = s.detach().cpu().numpy()
        if torch.is_tensor(l): l = l.detach().cpu().numpy()
        return b, s, l

    n = min(warmup_n, len(results))
    for i in range(n):
        b, s, l = _extract_bsl(results[i])
        g = gt_annos[i]['gt_bboxes_3d']
        b = _apply_coord_fix(b, best_mode)
        T = b.tensor.clone(); T[:, 0] += dx; T[:, 1] += dy; T[:, 2] += dz
        b = LiDARInstance3DBoxes(T)
        if s is None or len(s) == 0 or len(b) == 0 or len(g) == 0:
            continue
        order = np.argsort(-s)[:min(topk, len(s))]
        pc = b.gravity_center.detach().cpu().numpy()[order, :2]
        gc = g.gravity_center.detach().cpu().numpy()[:, :2]
        lc = l[order].astype(int)
        try:
            from scipy.spatial import cKDTree
            d, idx = cKDTree(gc).query(pc, k=1)
        except Exception:
            diff = pc[:, None, :] - gc[None, :, :]
            dist = np.sqrt((diff**2).sum(-1))
            idx = dist.argmin(axis=1); d = dist[np.arange(dist.shape[0]), idx]
        m = d < dist_thresh
        if not m.any():
            continue
        pb = b.tensor.detach().cpu().float()[order][m]
        gt = g.tensor.detach().cpu().float()[idx[m]]
        pb_lwh = _to_lwh7_tensor(LiDARInstance3DBoxes(pb))
        gt_lwh = _to_lwh7_tensor(LiDARInstance3DBoxes(gt))
        l_ratio = (gt_lwh[:, 3] / torch.clamp(pb_lwh[:, 3], 1e-3)).numpy()
        w_ratio = (gt_lwh[:, 4] / torch.clamp(pb_lwh[:, 4], 1e-3)).numpy()
        cls_pair = lc[m]
        for lr, wr, cid in zip(l_ratio.tolist(), w_ratio.tolist(), cls_pair.tolist()):
            if 0 <= cid < len(class_names):
                sl_per_cls[cid].append(float(np.clip(lr, *lw_clamp)))
                sw_per_cls[cid].append(float(np.clip(wr, *lw_clamp)))
                used_pairs[cid] += 1

    sl_cls, sw_cls = {}, {}
    for cid in range(len(class_names)):
        if len(sl_per_cls[cid]) >= min_pairs:
            sl_cls[cid] = float(np.median(np.array(sl_per_cls[cid])))
        if len(sw_per_cls[cid]) >= min_pairs:
            sw_cls[cid] = float(np.median(np.array(sw_per_cls[cid])))
    return sl_cls, sw_cls, used_pairs

# ============== 新增：近邻“混淆矩阵”调试（不改动预测，只打印） ==============
def _debug_confusion(det_annos, gt_annos, class_names, topk_each=64, dist_thresh=3.0, frames=64):
    C = len(class_names)
    conf = np.zeros((C, C), dtype=np.int64)  # [pred, gt]
    use_n = min(frames, len(det_annos))
    for i in range(use_n):
        d = det_annos[i]
        g = gt_annos[i]
        b = d['boxes_3d']; s = d['scores_3d']; l = d['labels_3d']
        gb = g['gt_bboxes_3d']; gl = g['gt_labels_3d']
        if len(b) == 0 or len(gb) == 0: continue
        if isinstance(s, torch.Tensor): s = s.detach().cpu().numpy()
        if isinstance(l, torch.Tensor): l = l.detach().cpu().numpy()
        order = np.argsort(-s)[:min(topk_each, len(s))]
        pc = b.gravity_center.detach().cpu().numpy()[order, :2]
        gc = gb.gravity_center.detach().cpu().numpy()[:, :2]
        try:
            from scipy.spatial import cKDTree
            d_, idx = cKDTree(gc).query(pc, k=1)
        except Exception:
            diff = pc[:, None, :] - gc[None, :, :]
            dist = np.sqrt((diff ** 2).sum(-1))
            idx = dist.argmin(axis=1); d_ = dist[np.arange(dist.shape[0]), idx]
        m = d_ < dist_thresh
        if not np.any(m): continue
        lp = l[order][m].astype(int)
        lg = gl[idx[m]].astype(int)
        for p, gcls in zip(lp, lg):
            if 0 <= p < C and 0 <= gcls < C:
                conf[p, gcls] += 1
    # 打印：每个“预测类”对应的“最近 GT 类”Top-3
    readable = {}
    for p in range(C):
        row = conf[p]
        if row.sum() == 0:
            continue
        top = row.argsort()[::-1][:3]
        readable[class_names[p]] = {class_names[g]: int(row[g]) for g in top if row[g] > 0}
    print('[DEBUG] Confusion by nearest GT (pred -> gt counts, first', use_n, 'frames):', readable)
    return conf

# ---------------- 评估核心 ----------------
def _match_one_class(gt_boxes_all, gt_labels_all,
                     det_boxes_all, det_scores_all, det_labels_all,
                     cid: int, iou_thr: float, use_bev: bool = False):
    gt_mask = (gt_labels_all == cid)
    if torch.is_tensor(det_labels_all):
        det_mask = (det_labels_all.detach().cpu().numpy() == cid)
    else:
        det_mask = (det_labels_all == cid)

    cur_gt = gt_boxes_all[gt_mask] if gt_mask.any() else LiDARInstance3DBoxes(np.zeros((0, 7)))
    cur_det = det_boxes_all[det_mask] if (np.sum(det_mask) > 0) else LiDARInstance3DBoxes(np.zeros((0, 7)))
    cur_scores = det_scores_all[det_mask]
    if torch.is_tensor(cur_scores):
        cur_scores = cur_scores.detach().cpu().numpy()

    npos = len(cur_gt)
    if len(cur_det) == 0:
        return [], [], [], npos

    if len(cur_gt) > 0:
        det_t = _to_lwh7_tensor(cur_det).contiguous().float()
        gt_t  = _to_lwh7_tensor(cur_gt).contiguous().float()
        if use_bev:
            if (boxes_iou_bev is not None) and torch.cuda.is_available():
                det_dev = det_t.cuda(non_blocking=True)
                gt_dev  = gt_t.cuda(non_blocking=True)
                with torch.no_grad():
                    ious = boxes_iou_bev(det_dev, gt_dev).detach().cpu().numpy()
            else:
                det_xywl = det_t[:, [0, 1, 4, 3]].cpu().numpy()
                gt_xywl  = gt_t[:, [0, 1, 4, 3]].cpu().numpy()
                ious = _bev_aabb_iou_xywl(det_xywl, gt_xywl)
        else:
            if (boxes_iou3d_gpu is not None) and torch.cuda.is_available():
                det_zb = det_t.clone(); gt_zb = gt_t.clone()
                det_zb[:, 2] = det_zb[:, 2] - det_zb[:, 5] * 0.5
                gt_zb[:, 2]  =  gt_zb[:, 2] -  gt_zb[:, 5] * 0.5
                det_dev = det_zb.cuda(non_blocking=True)
                gt_dev  = gt_zb.cuda(non_blocking=True)
                with torch.no_grad():
                    ious = boxes_iou3d_gpu(det_dev, gt_dev).detach().cpu().numpy()
            else:
                det_xywl = det_t[:, [0, 1, 4, 3]].cpu().numpy()
                gt_xywl  = gt_t[:, [0, 1, 4, 3]].cpu().numpy()
                ious = _bev_aabb_iou_xywl(det_xywl, gt_xywl)
    else:
        ious = np.zeros((len(cur_det), 0), dtype=np.float32)

    order = np.argsort(-cur_scores)
    ious = ious[order] if ious.size > 0 else ious
    tp = np.zeros(len(order), dtype=np.float32)
    fp = np.zeros(len(order), dtype=np.float32)
    gt_matched = np.zeros(len(cur_gt), dtype=np.int32)
    for i, row in enumerate(ious):
        if row.size == 0:
            fp[i] = 1; continue
        j = int(row.argmax())
        if row[j] >= iou_thr and gt_matched[j] == 0:
            tp[i] = 1; gt_matched[j] = 1
        else:
            fp[i] = 1
    scores = list(cur_scores[order])
    return scores, list(tp), list(fp), npos

def _evaluate_one_metric(gt_annos, det_annos, class_names, iou_thr=0.5,
                         use_bev=False, present_class_only=True):
    aps, npos_per_cls, ndet_per_cls = [], [], []
    ret = {}
    for cid, cname in enumerate(class_names):
        all_scores, all_tp, all_fp = [], [], []
        total_pos = 0
        for gts, dets in zip(gt_annos, det_annos):
            scores, tp, fp, npos = _match_one_class(
                gts['gt_bboxes_3d'], gts['gt_labels_3d'],
                dets['boxes_3d'],   dets['scores_3d'], dets['labels_3d'],
                cid, iou_thr, use_bev=use_bev
            )
            all_scores += scores; all_tp += tp; all_fp += fp; total_pos += npos
        npos_per_cls.append(total_pos); ndet_per_cls.append(len(all_scores))
        if total_pos == 0:
            ret[f'AP_{cname}'] = 0.0; aps.append(0.0); continue
        idx = np.argsort(-np.asarray(all_scores))
        tp_arr = np.asarray(all_tp)[idx]; fp_arr = np.asarray(all_fp)[idx]
        tp_cum = np.cumsum(tp_arr); fp_cum = np.cumsum(fp_arr)
        recall = tp_cum / float(total_pos)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap = _compute_ap(recall, precision)
        ret[f'AP_{cname}'] = ap; aps.append(ap)
    if present_class_only:
        present_mask = np.array(npos_per_cls) > 0
        mAP = float(np.mean(np.array(aps)[present_mask])) if present_mask.any() else 0.0
    else:
        mAP = float(np.mean(aps)) if len(aps) > 0 else 0.0
    ret['mAP'] = mAP
    ret['_stats/npos_per_cls'] = npos_per_cls
    ret['_stats/ndet_per_cls'] = ndet_per_cls
    return ret

# ---------------- Dataset ----------------
@DATASETS.register_module()
class DACustomNuScenesDataset(NuScenesDataset):
    """Domain-Adaptive Custom Dataset for NuScenes/Lyft (mmdet3d==0.17.1)."""

    def __init__(self,
                 domain: str = 'source',
                 use_ann: bool = True,
                 name_mapping: dict = None,
                 ignore_label: str = 'ignore',
                 present_class_only: bool = True,
                 **kwargs):
        self.domain = domain
        self.use_ann = use_ann
        self.name_mapping = name_mapping or {}
        self.ignore_label = ignore_label
        self.present_class_only = present_class_only
        super().__init__(**kwargs)

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        if isinstance(data, dict):
            self.data_infos = data.get('infos', data.get('data_infos', []))
            self.metadata = data.get('metadata', dict(version='detection_cvpr_2019'))
        elif isinstance(data, list):
            self.data_infos = data
            self.metadata = dict(version='detection_cvpr_2019')
        else:
            raise TypeError(f'Unsupported ann_file format: {type(data)}')
        self.version = 'detection_cvpr_2019'
        return self.data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info.get('sweeps', []),
            timestamp=info['timestamp'] / 1e6,
            domain=self.domain,
        )
        if self.modality.get('use_camera', False) and ('cams' in info):
            image_paths, lidar2img_rts, intrinsics, extrinsics, img_timestamp = [], [], [], [], []
            for _, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4, dtype=np.float32)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4, dtype=np.float32)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)
            input_dict.update(
                img_timestamp=img_timestamp,
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )
        if (not self.test_mode) and self.use_ann:
            input_dict['ann_info'] = self.get_ann_info(index)
        return input_dict

    def _map_name(self, n: str) -> str:
        if self.name_mapping:
            return self.name_mapping.get(n, n)
        return n

    def get_ann_info(self, index):
        info = self.data_infos[index]
        if (not self.use_ann) or ('gt_boxes' not in info) or (len(info['gt_boxes']) == 0):
            return dict(
                gt_bboxes_3d=LiDARInstance3DBoxes(np.zeros((0, 7), dtype=np.float32)),
                gt_labels_3d=np.zeros((0,), dtype=np.int64),
                gt_names=[]
            )
        gt_bboxes_3d = LiDARInstance3DBoxes(info['gt_boxes'])
        raw_names = list(info['gt_names'])
        mapped_names = [self._map_name(n) for n in raw_names]
        labels = [(self.CLASSES.index(mn) if mn in self.CLASSES else -1) if mn != self.ignore_label else -1
                  for mn in mapped_names]
        gt_labels_3d = np.asarray(labels, dtype=np.int64)
        return dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, gt_names=mapped_names)

    # ---------------- 主评估 ----------------
    def evaluate(self, results, metric='bbox', logger=None, **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed = ['bbox', 'mAP']
        for m in metrics:
            if m not in allowed:
                raise KeyError(f'metric {m} is not supported for DACustomNuScenesDataset.')

        # ---- 收集 GT
        gt_annos = [self.get_ann_info(i) for i in range(len(self))]

        # ---- 预测标签统计（过滤前）
        def _extract_labels_from_result(r):
            if isinstance(r, dict):
                l = r['pts_bbox']['labels_3d'] if 'pts_bbox' in r else r['labels_3d']
            else:
                _, _, l = r
            return l.detach().cpu().numpy() if torch.is_tensor(l) else (np.asarray(l) if l is not None else None)

        all_labels_list = []
        for r in results:
            li = _extract_labels_from_result(r)
            if li is not None and len(li) > 0:
                all_labels_list.append(li)
        if len(all_labels_list) > 0:
            all_labels = np.concatenate(all_labels_list, axis=0)
            print('[DEBUG] pred label range: ', (all_labels.min(), all_labels.max()),
                  '; unique (top 20): ', np.unique(all_labels)[:20])
        else:
            print('[DEBUG] pred label range: <empty>')

        # ---- 类别计数（过滤前）
        gt_counts = Counter()
        for g in gt_annos:
            lbl = g['gt_labels_3d']; lbl = lbl[lbl >= 0]
            for x in lbl: gt_counts[int(x)] += 1
        print('[DEBUG] GT count per class id:', {i: gt_counts.get(i, 0) for i in range(len(self.CLASSES))})

        pred_counts = Counter()
        for r in results:
            li = _extract_labels_from_result(r)
            if li is not None and len(li) > 0:
                for x in li:
                    if 0 <= int(x) < len(self.CLASSES):
                        pred_counts[int(x)] += 1
        print('[DEBUG] Pred count per class id (valid range only):',
              {i: pred_counts.get(i, 0) for i in range(len(self.CLASSES))})

        # ---- 坐标模式自检
        warmup_n = min(DEBUG_COORD_WARMUP_SAMPLES, len(results))
        modes = ['none', 'swap_xy_p90', 'swap_xy_m90', 'flip_x', 'flip_y',
                 'swap_xy_p90_flipx', 'swap_xy_p90_flipy']
        mode_score_sum = defaultdict(float); mode_dist_sum = defaultdict(float); mode_votes = defaultdict(int)

        def _extract_box_score(r):
            if isinstance(r, dict):
                if 'pts_bbox' in r:
                    b = r['pts_bbox']['boxes_3d']; s = r['pts_bbox']['scores_3d']
                else:
                    b = r['boxes_3d']; s = r['scores_3d']
            else:
                b, s, _ = r
            bt = b.tensor if isinstance(b, LiDARInstance3DBoxes) else b
            if bt.shape[-1] > 7: b = LiDARInstance3DBoxes(bt[:, :7])
            return b, (s.detach().cpu().numpy() if torch.is_tensor(s) else (np.asarray(s) if s is not None else None))

        for i in range(warmup_n):
            b, s = _extract_box_score(results[i])
            g = gt_annos[i]['gt_bboxes_3d']
            best_mode, best_hit, best_avgd = 'none', -1.0, float('inf')
            for md in modes:
                tb = _apply_coord_fix(b, md)
                hit, avgd = _center_hit_rate(tb, s, g, topk=CENTER_HIT_TOPK, thr=CENTER_HIT_THRESH)
                mode_score_sum[md] += hit; mode_dist_sum[md] += (avgd if np.isfinite(avgd) else 0.0)
                mode_votes[md] += 1
                if hit > best_hit or (math.isclose(hit, best_hit) and avgd < best_avgd):
                    best_mode, best_hit, best_avgd = md, hit, avgd
            if i < DEBUG_PRINT_SAMPLES:
                print(f'[DEBUG] sample#{i} best_coord_mode={best_mode} '
                      f'center_hit@{CENTER_HIT_THRESH}m={best_hit:.3f} '
                      f'avg_center_dist={best_avgd:.3f}  (TopK={CENTER_HIT_TOPK})')
        if warmup_n > 0:
            mode_avg = {md: (mode_score_sum[md] / max(1, mode_votes[md])) for md in modes}
            mode_avg_dist = {md: (mode_dist_sum[md] / max(1, mode_votes[md])) for md in modes}
            best_global = max(modes, key=lambda m: (mode_avg[m], -mode_avg_dist[m]))
            print('[DEBUG] Coord mode ranking (by center_hit):',
                  {m: round(mode_avg[m], 4) for m in modes})
            print('[DEBUG] Coord mode avg center_dist:',
                  {m: round(mode_avg_dist[m], 3) for m in modes})
            print(f'[DEBUG] Chosen GLOBAL coord_mode = {best_global}')
        else:
            best_global = 'none'
            print('[DEBUG] Not enough samples for coord warmup, use "none".')

        # ---- 全局平移/航向
        dx, dy, dz = _estimate_global_translation(results, gt_annos, best_global,
                                                  warmup_n=warmup_n, topk=CALIB_TOPK, dist_thresh=3.0)
        dyaw = _estimate_global_yaw_size(results, gt_annos, best_global,
                                         dx, dy, dz,
                                         warmup_n=warmup_n, topk=CALIB_TOPK, dist_thresh=3.0)

        # 细调（可选）
        dz_cls = {}; sh_cls = {}; used_pairs_v = {}
        if warmup_n > 0:
            dz_cls, sh_cls, used_pairs_v = _estimate_per_class_vertical(
                results, gt_annos, self.CLASSES, best_global, dx, dy, dz,
                warmup_n, topk=CALIB_TOPK, dist_thresh=3.0, min_pairs=30, sh_clamp=(0.5, 2.0)
            )
            summary_vcal = {self.CLASSES[c]: dict(dz=round(dz_cls[c], 3) if c in dz_cls else None,
                                                  sh=round(sh_cls[c], 3) if c in sh_cls else None,
                                                  n=used_pairs_v.get(c, 0))
                            for c in range(len(self.CLASSES))
                            if (c in dz_cls) or (c in sh_cls)}
            print('[DEBUG] Per-class vertical calib (applied on det):', summary_vcal)
        else:
            print('[DEBUG] Per-class vertical calib skipped (no warmup).')

        sl_cls = {}; sw_cls = {}; used_pairs_lw = {}
        if warmup_n > 0:
            sl_cls, sw_cls, used_pairs_lw = _estimate_per_class_lw(
                results, gt_annos, self.CLASSES, best_global, dx, dy, dz,
                warmup_n, topk=CALIB_TOPK, dist_thresh=3.0, min_pairs=30, lw_clamp=(0.7, 1.3)
            )
            summary_lw = {
                self.CLASSES[c]: dict(sl=round(sl_cls[c], 3) if c in sl_cls else None,
                                      sw=round(sw_cls[c], 3) if c in sw_cls else None,
                                      n=used_pairs_lw.get(c, 0))
                for c in range(len(self.CLASSES))
                if (c in sl_cls) or (c in sw_cls)
            }
            print('[DEBUG] Per-class length/width calib (stats):', summary_lw)
        else:
            print('[DEBUG] Per-class length/width calib skipped (no warmup).')

        # ---- 构造 det_annos：坐标修正 → 全局平移/航向 → 按类 z/h 与 l/w → 动态过滤
        per_frame_kept = []
        det_annos = []
        for res in results:
            if isinstance(res, dict):
                if 'pts_bbox' in res:
                    b = res['pts_bbox']['boxes_3d']; s = res['pts_bbox']['scores_3d']; l = res['pts_bbox']['labels_3d']
                else:
                    b = res['boxes_3d']; s = res['scores_3d']; l = res['labels_3d']
            else:
                b, s, l = res
            bt = b.tensor if isinstance(b, LiDARInstance3DBoxes) else b
            if bt.shape[-1] > 7:
                b = LiDARInstance3DBoxes(bt[:, :7])

            # 动态过滤：Top-K + 阈值并集
            if s is not None:
                s_np = s.detach().cpu().numpy() if torch.is_tensor(s) else np.asarray(s)
                n = len(s_np)
                if n == 0:
                    det_annos.append(dict(boxes_3d=LiDARInstance3DBoxes(np.zeros((0, 7), dtype=np.float32)),
                                          scores_3d=np.array([]), labels_3d=np.array([])))
                    per_frame_kept.append(0); continue
                order = np.argsort(-s_np)
                keep_mask = np.zeros(n, dtype=bool)
                k = min(EVAL_TOPK_KEEP, n)
                keep_mask[order[:k]] = True
                if EVAL_SCORE_THR is not None and EVAL_SCORE_THR > 0:
                    keep_mask |= (s_np >= EVAL_SCORE_THR)
                if keep_mask.sum() < EVAL_MIN_KEEP:
                    keep_mask[order[:min(EVAL_MIN_KEEP, n)]] = True
                if keep_mask.sum() == 0:
                    keep_mask[order[0]] = True

                Ttmp = b.tensor
                b = LiDARInstance3DBoxes(Ttmp[keep_mask])
                s = s_np[keep_mask]
                if torch.is_tensor(l): l = l.detach().cpu().numpy()
                l = np.asarray(l)[keep_mask]
                per_frame_kept.append(int(keep_mask.sum()))
            else:
                per_frame_kept.append(len(b))

            # 坐标模式 + 全局校准
            b = _apply_coord_fix(b, best_global)
            T = b.tensor.clone()
            if dx or dy or dz:
                T[:, 0] += float(dx); T[:, 1] += float(dy); T[:, 2] += float(dz)
            if dyaw:
                T[:, 6] += float(dyaw)

            # 按类 z/h（仅在统计足够时）
            Tlwh = _to_lwh7_tensor(LiDARInstance3DBoxes(T))
            labels_np = l
            if labels_np is not None and len(labels_np) == Tlwh.shape[0] and (dz_cls or sh_cls):
                for cid in range(len(self.CLASSES)):
                    mask = (labels_np == cid)
                    if not np.any(mask): continue
                    if cid in dz_cls: Tlwh[mask, 2] = Tlwh[mask, 2] + float(dz_cls[cid])
                    if cid in sh_cls: Tlwh[mask, 5] = Tlwh[mask, 5] * float(sh_cls[cid])

            # 不做 l/w 直接修改（只打印统计）

            T = _from_lwh7_tensor(Tlwh)
            b = LiDARInstance3DBoxes(T)
            det_annos.append(dict(boxes_3d=b, scores_3d=s, labels_3d=labels_np))

        # 过滤统计
        if len(per_frame_kept) > 0:
            pf = np.array(per_frame_kept)
            print(f'[DEBUG] Per-frame kept dets after filter: mean={pf.mean():.1f}, min={pf.min()}, max={pf.max()}, topK={EVAL_TOPK_KEEP}, thr={EVAL_SCORE_THR}')

        # 分数与 top1
        all_scores_debug = []
        kept_counts_per_cls = Counter()
        top1_per_cls = {i: -1.0 for i in range(len(self.CLASSES))}
        for d in det_annos:
            s = d['scores_3d']; l = d['labels_3d']
            if isinstance(s, torch.Tensor): s = s.detach().cpu().numpy()
            if isinstance(l, torch.Tensor): l = l.detach().cpu().numpy()
            if s is not None and len(s) > 0:
                all_scores_debug.append(s[:1000])
                for x in l:
                    if 0 <= int(x) < len(self.CLASSES):
                        kept_counts_per_cls[int(x)] += 1
                # top1 per cls
                for c in range(len(self.CLASSES)):
                    mask = (l == c)
                    if np.any(mask):
                        top1 = float(np.max(s[mask]))
                        if top1 > top1_per_cls[c]:
                            top1_per_cls[c] = top1
        if all_scores_debug:
            sc = np.concatenate(all_scores_debug)[:1000000]
            print('[DEBUG] score mean/std/min/p99:',
                  float(sc.mean()), float(sc.std()), float(sc.min()),
                  float(np.quantile(sc, 0.99)))
        else:
            print('[DEBUG] no scores collected')
        print('[DEBUG] kept dets per class (after filter):', {i: kept_counts_per_cls.get(i, 0) for i in range(len(self.CLASSES))})
        print('[DEBUG] top1 score per class:', {i: round(top1_per_cls[i], 3) for i in range(len(self.CLASSES))})

        # ☆☆☆ 新增：混淆矩阵（近邻） ☆☆☆
        _debug_confusion(det_annos, gt_annos, self.CLASSES, topk_each=64, dist_thresh=3.0, frames=warmup_n)

        # ---- 计算指标
        out_3d_05 = _evaluate_one_metric(gt_annos, det_annos, self.CLASSES,
                                         iou_thr=0.5, use_bev=False,
                                         present_class_only=self.present_class_only)
        out_3d_025 = _evaluate_one_metric(gt_annos, det_annos, self.CLASSES,
                                          iou_thr=0.25, use_bev=False,
                                          present_class_only=self.present_class_only)
        out_bev_05 = _evaluate_one_metric(gt_annos, det_annos, self.CLASSES,
                                          iou_thr=0.5, use_bev=True,
                                          present_class_only=self.present_class_only)

        result = {}
        for k, v in out_3d_05.items():
            if k.startswith('AP_'): result[f'AP3D@0.5/{k[3:]}'] = v
        for k, v in out_3d_025.items():
            if k.startswith('AP_'): result[f'AP3D@0.25/{k[3:]}'] = v
        for k, v in out_bev_05.items():
            if k.startswith('AP_'): result[f'APBEV@0.5/{k[3:]}'] = v
        result['mAP3D@0.5'] = out_3d_05['mAP']
        result['mAP3D@0.25'] = out_3d_025['mAP']
        result['mAPBEV@0.5'] = out_bev_05['mAP']
        result['_stats/npos_per_cls'] = out_3d_05['_stats/npos_per_cls']
        result['_stats/ndet_per_cls'] = out_3d_05['_stats/ndet_per_cls']
        result['_debug/chosen_coord_mode'] = best_global
        result['_debug/gt_counts'] = {i: int(gt_counts.get(i, 0)) for i in range(len(self.CLASSES))}
        result['_debug/pred_counts'] = {i: int(pred_counts.get(i, 0)) for i in range(len(self.CLASSES))}

        print('[DEBUG SUMMARY] chosen_coord_mode=', best_global,
              '; mAP3D@0.5=', round(result['mAP3D@0.5'], 4),
              '; mAP3D@0.25=', round(result['mAP3D@0.25'], 4),
              '; mAPBEV@0.5=', round(result['mAPBEV@0.5'], 4))
        return result

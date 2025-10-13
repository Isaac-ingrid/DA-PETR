# ------------------------------------------------------------------------
# PETR Head with:
#  - BEV branch
#  - QAL (query/image domain adversarial alignment)  [safe: detach/gate/clamp]
#  - QST (query-based self training with EMA thresholds) [nan-safe]
#  - 2D Image-View Decoder + Multi-label Head (auxiliary)
# ------------------------------------------------------------------------
import math
import numpy as np
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from mmcv.cnn import Conv2d, Linear, bias_init_with_prob
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply, reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet.models.utils import NormedLinear
from copy import deepcopy
from mmdet3d.core.bbox import LiDARInstance3DBoxes, DepthInstance3DBoxes, CameraInstance3DBoxes


# ---------- helpers ----------
def safe_softmax(x, dim=-1, eps=1e-6):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    p = F.softmax(x, dim=dim)
    return p.clamp(min=eps, max=1.0 - eps)


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


# ---------------- GRL ----------------
class _GradientReverseFn(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg().mul(ctx.lambd), None


class GradientReverse(nn.Module):
    def __init__(self, initial_lambda: float = 0.0):
        super().__init__()
        self._lambda = float(initial_lambda)

    def set_lambda(self, lambd: float):
        self._lambda = float(lambd)

    def forward(self, x):
        return _GradientReverseFn.apply(x, self._lambda)


# ----------- A tiny Transformer decoder for 2D image-view branch -----------
class SimpleDecoderLayer(nn.Module):
    """Minimal Transformer decoder layer (self-attn + cross-attn + FFN)."""
    def __init__(self, embed_dims=256, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dims)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, mem, mem_key_padding_mask=None, pos=None):
        # self-attn
        q2, _ = self.self_attn(q, q, q)
        q = self.norm1(q + self.dropout(q2))
        # cross-attn (pos added to mem)
        k = v = mem if pos is None else (mem + pos)
        q2, _ = self.cross_attn(q, k, v, key_padding_mask=mem_key_padding_mask)
        q = self.norm2(q + self.dropout(q2))
        # ffn
        q2 = self.ffn(q)
        q = self.norm3(q + self.dropout(q2))
        return q


class SimpleDecoder(nn.Module):
    def __init__(self, num_layers=3, embed_dims=256, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleDecoderLayer(embed_dims, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, q, mem, mem_mask=None, pos=None):
        for l in self.layers:
            q = l(q, mem, mem_key_padding_mask=mem_mask, pos=pos)
        return q


@HEADS.register_module()
class DAPETRHead(AnchorFreeHead):
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(type='SinePositionalEncoding', num_feats=128, normalize=True),
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(type='CrossEntropyLoss', bg_cls_weight=0.1, use_sigmoid=False, loss_weight=1.0, class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(assigner=dict(type='HungarianAssigner',
                                              cls_cost=dict(type='ClassificationCost', weight=1.),
                                              reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                                              iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 with_position=True,
                 with_multiview=False,
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start=1,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 init_cfg=None,
                 normedlinear=False,
                 # ==== DA / QAL / QST ====
                 da_cfg: Dict[str, Any] = None,
                 query_init_cfg: Dict[str, Any] = None,
                 qst_cfg: Dict[str, Any] = None,
                 # ==== BEV branch ====
                 num_bev_query: int = 900,
                 bev_num_layers: int = 6,
                 use_separate_bev_decoder: bool = False,
                 # ==== 2D Image-View decoder + head ====
                 img_view_decoder: Dict[str, Any] = None,
                 img_view_head: Dict[str, Any] = None,
                 **kwargs):
        self.num_classes = num_classes
        self.in_channels = in_channels

        # code size / weights
        self.code_size = int(kwargs.get('code_size', 10))
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[:self.code_size]

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DAPETRHead):
            assert isinstance(class_weight, float)
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            class_weight = torch.ones(num_classes + 1) * class_weight
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight']
            assert loss_bbox['loss_weight'] == assigner['reg_cost']['weight']
            self.assigner = build_assigner(assigner)
            self.sampler = build_sampler(dict(type='PseudoSampler'), context=self)

        self.num_query = num_query
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = 256
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = 0
        self.with_position = with_position
        self.with_multiview = with_multiview

        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims

        self.num_pred = 6
        self.normedlinear = normedlinear

        super(DAPETRHead, self).__init__(num_classes, in_channels, init_cfg=init_cfg)

        # losses
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.cls_out_channels = num_classes if self.loss_cls.use_sigmoid else (num_classes + 1)

        # PE / Transformer / Coder
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.aux_dec_weight = float(getattr(self, 'aux_dec_weight', 0.5))

        self._init_layers()

        # ===== DA/QAL =====
        self.da_cfg = da_cfg or {}
        self._da_detach = bool((self.da_cfg or {}).get('disc_detach', True))
        self.hidden_dim = self.embed_dims
        _grl_cfg = self.da_cfg.get('grl', {}) or {}
        self.grl = GradientReverse(initial_lambda=float(_grl_cfg.get('init', 0.0)))
        self._grl_lambda_init = float(_grl_cfg.get('init', 0.0))
        self._grl_lambda_max = float(_grl_cfg.get('max', 1.0))
        self._grl_warmup_ep = int(_grl_cfg.get('warmup_epochs', 5))
        self._grl_force_max = bool((self.da_cfg.get('grl', {}) or {}).get('force_max', False))

        # loss weights（兼容 query_da → bev_da）
        lw = (self.da_cfg.get('loss_weights', {}) if isinstance(self.da_cfg, dict) else {})
        self._w_img = float(lw.get('img_da', 0.0))
        self._w_bev = float(lw.get('bev_da', lw.get('query_da', 0.0)))

        # 判别器工厂
        def _make_disc(in_dim, hid_dim=256, num_layers=2, dropout=0.1, out_dim=2):
            layers = []
            for i in range(num_layers):
                layers += [
                    nn.Linear(in_dim if i == 0 else hid_dim, hid_dim),
                    nn.ReLU(inplace=True)
                ]
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers += [nn.Linear(hid_dim, out_dim)]
            return nn.Sequential(*layers)

        # 构建 img / bev 判别器（优先使用 da_cfg.img_disc / bev_disc；否则若开启标志或权重大于 0 也自动建一个默认的）
        img_disc_cfg = (self.da_cfg.get('img_disc', None) or None)
        bev_disc_cfg = (self.da_cfg.get('bev_disc', None) or None)

        need_img_disc = (img_disc_cfg is not None) or bool(self.da_cfg.get('use_img_da', False)) or (self._w_img > 0)
        need_bev_disc = (bev_disc_cfg is not None) or bool(self.da_cfg.get('use_query_da', False)) or (self._w_bev > 0)

        if need_img_disc:
            kw = dict(in_dim=self.hidden_dim, hid_dim=self.hidden_dim, num_layers=2, dropout=0.1)
            if isinstance(img_disc_cfg, dict):
                kw.update(img_disc_cfg)
            self.img_da_disc = _make_disc(**kw)
            self.loss_da_img = nn.CrossEntropyLoss()
        else:
            self.img_da_disc, self.loss_da_img = None, None

        if need_bev_disc:
            kw = dict(in_dim=self.hidden_dim, hid_dim=self.hidden_dim, num_layers=2, dropout=0.1)
            if isinstance(bev_disc_cfg, dict):
                kw.update(bev_disc_cfg)
            self.bev_da_disc = _make_disc(**kw)
            self.loss_da_bev = nn.CrossEntropyLoss()
        else:
            self.bev_da_disc, self.loss_da_bev = None, None

        # ===== QST =====
        self.qst_cfg = qst_cfg or {}
        self.qst_enable = bool(self.qst_cfg.get('enable', False))
        self.qst_queue_len = int(self.qst_cfg.get('queue_len', 50))
        self.qst_ema_gamma = float(self.qst_cfg.get('ema_gamma', 0.9))
        self.qst_top_percent = float(self.qst_cfg.get('top_percent', 0.2))
        self.qst_loss_w = float(self.qst_cfg.get('loss_weight', 0.5))
        self.qst_use_det_loss = bool(self.qst_cfg.get('use_det_loss', False))
        self.register_buffer('qst_mu', torch.zeros(self.num_classes))
        self.register_buffer('qst_sigma2', torch.ones(self.num_classes) * 0.05)
        self.register_buffer('qst_hist', torch.zeros(self.qst_queue_len, self.num_classes))
        self.qst_ptr = 0

        # ===== BEV branch =====
        self.num_bev_query = num_bev_query
        self.bev_num_layers = bev_num_layers
        self.use_separate_bev_decoder = use_separate_bev_decoder

        self.bev_reference_points = nn.Embedding(self.num_bev_query, 3)
        nn.init.uniform_(self.bev_reference_points.weight.data, 0, 1)

        self.bev_query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims), nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims)
        )
        self.bev_transformer = deepcopy(self.transformer)

        def _disable_cp(m):
            if hasattr(m, 'with_cp'):
                m.with_cp = False

        # 两套 transformer 都关 gradient checkpoint，避免 DDP/重入冲突
        self.transformer.apply(_disable_cp)
        self.bev_transformer.apply(_disable_cp)

        bev_cls_branch, bev_reg_branch = [], []
        for _ in range(self.num_reg_fcs):
            bev_cls_branch += [Linear(self.embed_dims, self.embed_dims), nn.LayerNorm(self.embed_dims), nn.ReLU(inplace=True)]
        bev_cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        for _ in range(self.num_reg_fcs):
            bev_reg_branch += [Linear(self.embed_dims, self.embed_dims), nn.ReLU()]
        bev_reg_branch.append(Linear(self.embed_dims, self.code_size))
        self._bev_cls_proto = nn.Sequential(*bev_cls_branch)
        self._bev_reg_proto = nn.Sequential(*bev_reg_branch)
        self.bev_cls_branches = nn.ModuleList([deepcopy(self._bev_cls_proto) for _ in range(self.bev_num_layers)])
        self.bev_reg_branches = nn.ModuleList([deepcopy(self._bev_reg_proto) for _ in range(self.bev_num_layers)])

        # ===== 2D Image-View decoder + head =====
        self.use_aux_2d = bool((self.da_cfg or {}).get('use_aux_2d_head', False) or (img_view_decoder is not None))
        if self.use_aux_2d:
            ivd = img_view_decoder or {}
            self.iv_num_layers = int(ivd.get('num_layers', 3))
            self.iv_num_queries = int(ivd.get('num_queries', 300))
            self.iv_embed_dims = int(ivd.get('embed_dims', self.embed_dims))
            self.iv_num_heads = int(ivd.get('num_heads', 8))
            self.iv_ffn_dim = int(ivd.get('ffn_dim', 1024))
            self.iv_dropout = float(ivd.get('dropout', 0.1))

            self.iv_queries = nn.Embedding(self.iv_num_queries, self.iv_embed_dims)
            nn.init.normal_(self.iv_queries.weight, std=0.02)

            self.iv_decoder = SimpleDecoder(
                num_layers=self.iv_num_layers,
                embed_dims=self.iv_embed_dims,
                num_heads=self.iv_num_heads,
                ffn_dim=self.iv_ffn_dim,
                dropout=self.iv_dropout
            )

            ivh = img_view_head or {}
            self.loss_iv = build_loss(ivh.get('loss_cls', dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)))
            self.iv_cls_head = nn.Sequential(
                nn.Linear(self.iv_embed_dims, self.iv_embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.iv_embed_dims, num_classes)
            )
        else:
            self.iv_decoder, self.iv_cls_head, self.loss_iv = None, None, None

        # 参考点/PE 稳健化
        self.query_init_cfg = query_init_cfg or {}
        self._freeze_pe_epochs = int(self.query_init_cfg.get('freeze_pe_epochs', 0))
        self._pe_noise_std = float(self.query_init_cfg.get('pe_noise_std', 0.0))

        self._qst_detach_inputs = bool((self.qst_cfg or {}).get('detach_inputs', True))
        self.bev_detach = bool((self.da_cfg or {}).get('bev_detach', True))  # 默认 True

    # ----- utils -----
    def update_grl_lambda(self, cur_epoch: int):
        # warmup -> max，并做硬上限
        if self._grl_warmup_ep <= 0:
            new_l = self._grl_lambda_max
        else:
            ratio = min(max(cur_epoch, 0) / float(self._grl_warmup_ep), 1.0)
            new_l = self._grl_lambda_init + (self._grl_lambda_max - self._grl_lambda_init) * ratio
        self.grl.set_lambda(float(min(max(new_l, 0.0), self._grl_lambda_max)))

    def _init_layers(self):
        self.input_proj = Conv2d(self.in_channels, self.embed_dims, kernel_size=1)

        # 主分支 head 原型
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch += [Linear(self.embed_dims, self.embed_dims), nn.LayerNorm(self.embed_dims), nn.ReLU(inplace=True)]
        cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels) if self.normedlinear
                          else Linear(self.embed_dims, self.cls_out_channels))
        fc_cls_proto = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch += [Linear(self.embed_dims, self.embed_dims), nn.ReLU()]
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_proto = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList([deepcopy(fc_cls_proto) for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList([deepcopy(reg_proto) for _ in range(self.num_pred)])

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims * 4, 1, 1, 0), nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims, 1, 1, 0))
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, 1, 1, 0), nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, 1, 1, 0))

        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims * 4, 1, 1, 0), nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims, 1, 1, 0))

        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims), nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims))

    def init_weights(self):
        self.transformer.init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
            for m in getattr(self, 'bev_cls_branches', []):
                try:
                    nn.init.constant_(m[-1].bias, bias_init)
                except Exception:
                    pass

    # ----- PE -----
    def position_embeding(self, img_feats, img_metas, masks=None):
        """构建 3D 位置编码（dtype/device 安全，meshgrid 指定 indexing）。"""
        eps = 1e-5
        feat = img_feats[self.position_level]
        B, N, C, H, W = feat.shape
        dev = feat.device
        dty = feat.dtype

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]

        coords_h = torch.arange(H, device=dev, dtype=dty) * (float(pad_h) / float(H))
        coords_w = torch.arange(W, device=dev, dtype=dty) * (float(pad_w) / float(W))

        index = torch.arange(0, self.depth_num, device=dev, dtype=dty)
        if self.LID:
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * (index + 1)
        else:
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        ww, hh, dd = torch.meshgrid(coords_w, coords_h, coords_d, indexing='ij')  # (W,H,D)
        coords = torch.stack([ww, hh, dd], dim=-1)  # (W,H,D,3)
        one = torch.ones_like(coords[..., :1])
        coords = torch.cat([coords, one], dim=-1)   # (W,H,D,4)
        depth = torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)
        coords[..., :2] = coords[..., :2] * depth

        # lidar2img -> img2lidar：转 float32，放到 dev/dtype
        img2lidars_np = []
        for img_meta in img_metas:
            mats = []
            for i in range(len(img_meta['lidar2img'])):
                mats.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars_np.append(np.asarray(mats, dtype=np.float32))
        img2lidars = torch.as_tensor(np.asarray(img2lidars_np, dtype=np.float32), device=dev, dtype=dty)  # (B,N,4,4)

        coords = coords.to(device=dev, dtype=dty)
        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)      # (B,N,W,H,D,4,1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]           # (B,N,W,H,D,3)

        x0, y0, z0, x1, y1, z1 = self.position_range
        coords3d[..., 0] = (coords3d[..., 0] - x0) / (x1 - x0)
        coords3d[..., 1] = (coords3d[..., 1] - y0) / (y1 - y0)
        coords3d[..., 2] = (coords3d[..., 2] - z0) / (z1 - z0)

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)           # (B,N,W,H,D,3)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)   # (B,N,W,H)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)       # (B,N,H,W)

        coords3d = coords3d.permute(0,1,4,5,3,2).contiguous().view(B * N, -1, H, W)  # (B*N, 3*D, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)                   # (B*N, C, H, W)
        coords_position_embeding = coords_position_embeding.view(B, N, self.embed_dims, H, W)

        return coords_position_embeding, coords_mask

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is DAPETRHead:
            convert_dict = {'.self_attn.': '.attentions.0.', '.multihead_attn.': '.attentions.1.', '.decoder.norm.': '.decoder.post_norm.'}
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori, conv in convert_dict.items():
                    if ori in k:
                        state_dict[k.replace(ori, conv)] = state_dict[k]
                        del state_dict[k]
        super(AnchorFreeHead, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    # ----- helper -----
    def _domains_from_img_metas(self, img_metas):
        ds = []
        for m in img_metas:
            d = m.get('domain', 'source')
            ds.append(0 if (isinstance(d, str) and d.lower().startswith('source')) else 1)
        return torch.tensor(ds, dtype=torch.long, device=self.reference_points.weight.device)

    def _qal_weights(self, logits, domains):
        if logits is None or domains is None:
            return None
        probs = safe_softmax(logits, dim=-1)
        p_src = probs[:, 0]
        mask_s = (domains == 0); mask_t = (domains == 1)
        eps = 1e-6
        ps = p_src[mask_s].mean() if mask_s.any() else p_src.mean()
        pt = p_src[mask_t].mean() if mask_t.any() else p_src.mean()
        log_ps = torch.log(ps.clamp(min=eps, max=1-eps))
        log_1mpt = torch.log((1-pt).clamp(min=eps, max=1-eps))
        raw = -(log_ps + log_1mpt)          # 越大 => 惩罚越强
        raw = raw.clamp(min=-5.0, max=5.0)  # 防爆
        lam = torch.exp(raw).clamp(min=0.5, max=5.0)
        return lam.detach()

    def _qst_update_queue(self, p_denoised_mean):  # (C,)
        if not torch.isfinite(p_denoised_mean).all():
            return
        p = p_denoised_mean.clamp(0.0, 1.0)
        self.qst_hist[self.qst_ptr % self.qst_queue_len] = p.detach()
        self.qst_ptr += 1
        B = min(self.qst_ptr, self.qst_queue_len)
        if B == 0:
            return
        hist = self.qst_hist[:B]
        mu_hat = hist.mean(dim=0)
        var_hat = hist.var(dim=0, unbiased=(B > 1))
        gamma = self.qst_ema_gamma
        mu_new = gamma * self.qst_mu + (1 - gamma) * mu_hat
        var_new = gamma * self.qst_sigma2 + (1 - gamma) * var_hat + 1e-6
        self.qst_mu = torch.nan_to_num(mu_new, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        self.qst_sigma2 = torch.nan_to_num(var_new, nan=0.05, posinf=0.25, neginf=1e-6).clamp_min(1e-6)

    def _qst_thresholds(self):
        if self.qst_ptr == 0:
            return torch.full_like(self.qst_mu, 0.5)
        from torch.distributions.normal import Normal
        mu = torch.nan_to_num(self.qst_mu, nan=0.5).clamp(0.0, 1.0)
        sigma = torch.sqrt(torch.nan_to_num(self.qst_sigma2, nan=0.05)).clamp_min(1e-4)
        dist = Normal(mu, sigma, validate_args=False)
        Phi1 = dist.cdf(torch.ones_like(mu))
        Phi0 = dist.cdf(torch.zeros_like(mu))
        v = self.qst_top_percent
        target_cdf = (Phi1 - v * (Phi1 - Phi0)).clamp(1e-6, 1-1e-6)
        tau = dist.icdf(target_cdf)
        return torch.nan_to_num(tau, nan=0.5).clamp(0.0, 1.0)

    # ----- forward -----
    def forward(self, mlvl_feats, img_metas):
        import torch
        import torch.nn.functional as F
        import torch.distributed as dist

        EPS = 1e-4
    
        # 1) 可选：强制 GRL 拉满（默认不拉满，交给调度器）
        if self.training and getattr(self, "_grl_force_max", False):
            self.grl.set_lambda(self._grl_lambda_max)

        # 2) 基础特征 & mask
        x = mlvl_feats[0]                     # (B,N,C,H,W)
        B, N = x.size(0), x.size(1)
        H0, W0, _ = img_metas[0]['pad_shape'][0]

        masks = x.new_ones((B, N, H0, W0))
        for b in range(B):
            for n in range(N):
                h, w, _ = img_metas[b]['img_shape'][n]
                masks[b, n, :h, :w] = 0

        x = self.input_proj(x.flatten(0, 1))  # (B*N,C',H',W')
        x = x.view(B, N, *x.shape[-3:])
        masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)

        # 3) 位置编码
        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks)
            pos_embed = coords_position_embeding
            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pes = [self.positional_encoding(masks[:, i, :, :]).unsqueeze(1) for i in range(N)]
                sin_embed = torch.cat(pes, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pes = [self.positional_encoding(masks[:, i, :, :]).unsqueeze(1) for i in range(N)]
                pos_embed = torch.cat(pes, 1)

        # 4) Image-view 主分支
        reference_points = self.reference_points.weight  # (Q,3) in [0,1]
        if self.training and getattr(self, "_pe_noise_std", 0.0) > 0:
            reference_points = reference_points + torch.randn_like(reference_points) * self._pe_noise_std
        reference_points = torch.clamp(reference_points, EPS, 1.0 - EPS)
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        reference_points_b = reference_points.unsqueeze(0).repeat(B, 1, 1)

        outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed, self.reg_branches)
        outs_dec = torch.nan_to_num(outs_dec)

        out_cls, out_reg = [], []
        reference_is = inverse_sigmoid(reference_points_b)
        for lvl in range(outs_dec.shape[0]):
            logits = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:2] = (tmp[..., 0:2] + reference_is[..., 0:2]).sigmoid()
            tmp[..., 4:5] = (tmp[..., 4:5] + reference_is[..., 2:3]).sigmoid()
            out_cls.append(logits); out_reg.append(tmp)

        all_cls_scores = torch.stack(out_cls)   # [L,B,Q,C]
        all_bbox_preds = torch.stack(out_reg)   # [L,B,Q,9]
        all_bbox_preds[..., 0:1] = all_bbox_preds[..., 0:1] * (self.pc_range[3]-self.pc_range[0]) + self.pc_range[0]
        all_bbox_preds[..., 1:2] = all_bbox_preds[..., 1:2] * (self.pc_range[4]-self.pc_range[1]) + self.pc_range[1]
        all_bbox_preds[..., 4:5] = all_bbox_preds[..., 4:5] * (self.pc_range[5]-self.pc_range[2]) + self.pc_range[2]

        # 5) 全局“混域”门控（各 rank 对 num_src/num_tgt 做 all_reduce，统一控制流）
        domains = self._domains_from_img_metas(img_metas)   # (B,)
        num_src_local = (domains == 0).sum().to(all_cls_scores.device, dtype=torch.int32)
        num_tgt_local = (domains == 1).sum().to(all_cls_scores.device, dtype=torch.int32)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_src_local, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(num_tgt_local, op=torch.distributed.ReduceOp.SUM)
        has_mixed_global = (num_src_local > 0) & (num_tgt_local > 0)
        da_gate = has_mixed_global.float().to(all_cls_scores.device)  # shape: []

        # 一个安全的“门控 logits”工具：gate==1 时正常反传；gate==0 时仅用 detached 分支，梯度为 0
        def gate_logits(logits, gate_scalar: torch.Tensor):
            g = gate_scalar
            while g.dim() < logits.dim():
                g = g.view(*g.shape, 1)
            return logits * g + logits.detach() * (1.0 - g)

        # 6) Image-DA（始终前向；仅在混域时产生梯度）
        da_img_logits, da_aux = None, {}
        if self.img_da_disc is not None:
            feat_bnchw = x.flatten(0, 1)
            if getattr(self, "_da_detach", True):
                feat_bnchw = feat_bnchw.detach()
            pooled = F.adaptive_avg_pool2d(feat_bnchw, 1).flatten(1).view(B, N, -1).mean(dim=1)  # (B,C)
            raw_img_logits = self.img_da_disc(self.grl(pooled))
            raw_img_logits = torch.nan_to_num(raw_img_logits, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-15, 15)
            da_img_logits = gate_logits(raw_img_logits, da_gate)
            da_aux['img_feat'] = pooled

        # 7) BEV 分支 & BEV-DA
        bev_ref = self.bev_reference_points.weight
        if self.training and getattr(self, "_pe_noise_std", 0.0) > 0:
            bev_ref = bev_ref + torch.randn_like(bev_ref) * self._pe_noise_std
        bev_ref = torch.clamp(bev_ref, EPS, 1.0 - EPS)
        bev_query = self.bev_query_embedding(pos2posemb3d(bev_ref))
        bev_ref_b = bev_ref.unsqueeze(0).repeat(B, 1, 1)

        if getattr(self, "use_separate_bev_decoder", False):
            x_bev = x.detach() if (self.training and getattr(self, "bev_detach", True)) else x
            bev_outs_dec, _ = self.bev_transformer(x_bev, masks, bev_query, pos_embed, self.bev_reg_branches)
        else:
            bev_outs_dec = outs_dec
        bev_outs_dec = torch.nan_to_num(bev_outs_dec)

        bev_cls_list, bev_reg_list = [], []
        bev_ref_is = inverse_sigmoid(bev_ref_b)
        for l in range(bev_outs_dec.shape[0]):
            logits = self.bev_cls_branches[l](bev_outs_dec[l])
            tmp = self.bev_reg_branches[l](bev_outs_dec[l])
            tmp[..., 0:2] = (tmp[..., 0:2] + bev_ref_is[..., 0:2]).sigmoid()
            tmp[..., 4:5] = (tmp[..., 4:5] + bev_ref_is[..., 2:3]).sigmoid()
            bev_cls_list.append(logits); bev_reg_list.append(tmp)

        bev_cls_scores = torch.stack(bev_cls_list)
        bev_bbox_preds = torch.stack(bev_reg_list)
        bev_bbox_preds[..., 0:1] = bev_bbox_preds[..., 0:1] * (self.pc_range[3]-self.pc_range[0]) + self.pc_range[0]
        bev_bbox_preds[..., 1:2] = bev_bbox_preds[..., 1:2] * (self.pc_range[4]-self.pc_range[1]) + self.pc_range[1]
        bev_bbox_preds[..., 4:5] = bev_bbox_preds[..., 4:5] * (self.pc_range[5]-self.pc_range[2]) + self.pc_range[2]

        # 主分支概率（用于 QST）
        p_iv_main  = safe_softmax(all_cls_scores[-1], dim=-1)[..., :self.num_classes]
        p_bev_main = safe_softmax(bev_cls_scores[-1], dim=-1)[..., :self.num_classes]
        if getattr(self, "_qst_detach_inputs", True):
            p_iv_main  = p_iv_main.detach()
            p_bev_main = p_bev_main.detach()

        da_bev_logits = None
        if self.bev_da_disc is not None:
            bev_last = bev_outs_dec[-1].mean(dim=1)  # (B,C)
            if getattr(self, "_da_detach", True):
                bev_last = bev_last.detach()
            raw_bev_logits = self.bev_da_disc(self.grl(bev_last))
            raw_bev_logits = torch.nan_to_num(raw_bev_logits, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-15, 15)
            da_bev_logits = gate_logits(raw_bev_logits, da_gate)

        # 8) 2D 辅助
        iv_logits = None
        if getattr(self, "use_aux_2d", False):
            B_, N_, C_, H_, W_ = x.shape
            mem = x.view(B_, N_, C_, H_ * W_).permute(0, 1, 3, 2).reshape(B_, N_ * H_ * W_, C_)
            mem_mask = masks.view(B_, N_, H_ * W_).reshape(B_, N_ * H_ * W_)
            pos = pos_embed.view(B_, N_, C_, H_, W_).permute(0, 1, 3, 4, 2).reshape(B_, N_ * H_ * W_, C_)
            q = self.iv_queries.weight.unsqueeze(0).repeat(B_, 1, 1)
            q = self.iv_decoder(q, mem, mem_mask, pos=pos)
            iv_logits = self.iv_cls_head(q)
            iv_logits = torch.nan_to_num(iv_logits, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-15, 15)

        # 9) 输出；把 da_gate 一并传给 loss
        outs = dict(
            all_cls_scores=all_cls_scores,
            all_bbox_preds=all_bbox_preds,
            enc_cls_scores=None,
            enc_bbox_preds=None,
            # DA
            da_img_logits=da_img_logits,
            da_bev_logits=da_bev_logits,
            da_gate=da_gate,
            domains=domains,
            da_aux={'img_feat': da_aux.get('img_feat', None)},
            # BEV
            bev_cls_scores=bev_cls_scores,
            bev_bbox_preds=bev_bbox_preds,
            # QST
            p_iv=p_iv_main,
            p_bev=p_bev_main,
            # 2D
            iv_logits=iv_logits
        )
        return outs


    # ----- targets & losses -----
    def _get_target_single(self, cls_score, bbox_pred, gt_labels, gt_bboxes, gt_bboxes_ignore=None):
        """稳健版：对齐 GT 到预测维度，并处理无正样本场景。"""
        num_bboxes = bbox_pred.size(0)
        pred_dim = bbox_pred.size(-1)

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # 分类 target
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        if pos_inds.numel() > 0:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # 回归 target（维度对齐到 pred_dim）
        bbox_targets = bbox_pred.new_zeros((num_bboxes, pred_dim))
        bbox_weights = bbox_pred.new_zeros((num_bboxes, pred_dim))
        if pos_inds.numel() > 0:
            gt_pos = sampling_result.pos_gt_bboxes  # (P, D_gt)
            D_gt = gt_pos.size(-1)
            if D_gt < pred_dim:
                pad = bbox_pred.new_zeros((gt_pos.size(0), pred_dim - D_gt))
                gt_pos = torch.cat([gt_pos, pad], dim=-1)
            elif D_gt > pred_dim:
                gt_pos = gt_pos[:, :pred_dim]
            bbox_targets[pos_inds] = gt_pos
            bbox_weights[pos_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets(self, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list, gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self, cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list=None):
        """稳健版：pred/target 维度自动对齐；空样本与 NaN 掩码兜底。"""
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        # (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg) = multi_apply(
        #     self.get_targets, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg) = self.get_targets(
            cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # 分类损失
        cls_scores_flat = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores_flat.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores_flat, labels, label_weights, avg_factor=cls_avg_factor)

        # 归一化与维度对齐
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        bbox_preds_flat = bbox_preds.reshape(-1, bbox_preds.size(-1))                  # [N, D_pred]
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)         # [N, D_tgt*]
        D_pred = bbox_preds_flat.size(-1)
        D_tgt = normalized_bbox_targets.size(-1)
        D_w = self.code_weights.numel()
        D_eff = min(D_pred, D_tgt, D_w)                                               # 有效对齐维度

        if D_tgt < D_pred:
            pad = bbox_preds_flat.new_zeros((normalized_bbox_targets.size(0), D_pred - D_tgt))
            normalized_bbox_targets = torch.cat([normalized_bbox_targets, pad], dim=-1)
        elif D_tgt > D_pred:
            normalized_bbox_targets = normalized_bbox_targets[:, :D_pred]

        isnotnan = torch.isfinite(normalized_bbox_targets[:, :D_eff]).all(dim=-1)
        if isnotnan.sum() == 0:
            loss_bbox = bbox_preds_flat.sum() * 0.0
        else:
            bbox_weights = bbox_weights * self.code_weights[:D_pred]
            loss_bbox = self.loss_bbox(
                bbox_preds_flat[isnotnan, :D_eff],
                normalized_bbox_targets[isnotnan, :D_eff],
                bbox_weights[isnotnan, :D_eff],
                avg_factor=num_total_pos
            )

        return torch.nan_to_num(loss_cls), torch.nan_to_num(loss_bbox)

    # ====== main loss + QAL + QST + 2D aux ======
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        da_img_logits = preds_dicts.get('da_img_logits', None)
        da_bev_logits = preds_dicts.get('da_bev_logits', None)
        domains = preds_dicts.get('domains', None)
        iv_logits = preds_dicts.get('iv_logits', None)

        # 被 force_fp32 转成 float 的 target 转回 long
        if domains is not None and domains.dtype != torch.long:
            domains = domains.long()

        # 组装 GT
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(device)
                          for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()
        if enc_cls_scores is not None:
            binary_labels_list = [torch.zeros_like(gt_labels_list[i]) for i in range(len(all_gt_labels_list))]
            enc_loss_cls, enc_losses_bbox = self.loss_single(enc_cls_scores, enc_bbox_preds, gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        # 中间层监督打折
        for i, (lc, lb) in enumerate(zip(losses_cls[:-1], losses_bbox[:-1])):
            loss_dict[f'd{i}.loss_cls']  = lc * self.aux_dec_weight
            loss_dict[f'd{i}.loss_bbox'] = lb * self.aux_dec_weight

        # ===== QAL / DA loss =====
        w_img = self._w_img
        w_bev = self._w_bev

        # 统一门控（若老模型没传 da_gate，就默认为 1）
        gate = preds_dicts.get('da_gate', None)
        if gate is None:
            device_gate = (da_img_logits if da_img_logits is not None else all_cls_scores[-1]).device
            gate = all_cls_scores[-1].sum().detach().new_tensor(1.0).to(device_gate)

        def _safe_da_ce(loss_fn, logits, labels):
            if logits is None or labels is None or logits.numel() == 0:
                return None
            if not torch.isfinite(logits).all():
                return None
            return loss_fn(logits, labels)

        # img-DA
        if w_img > 0 and self.loss_da_img is not None:
            l_img_raw = _safe_da_ce(self.loss_da_img, da_img_logits, domains)
            if l_img_raw is not None:
                loss_dict['loss_da_img_raw'] = l_img_raw.detach()
                lam_iv = self._qal_weights(da_img_logits, domains)
                l_img = (lam_iv * l_img_raw) if (lam_iv is not None) else l_img_raw
                loss_dict['loss_da_img'] = (l_img * gate) * w_img
            else:
                loss_dict['loss_da_img'] = all_cls_scores[-1].sum() * 0.0

        # bev-DA
        if w_bev > 0 and self.loss_da_bev is not None:
            l_bev_raw = _safe_da_ce(self.loss_da_bev, da_bev_logits, domains)
            if l_bev_raw is not None:
                loss_dict['loss_da_bev_raw'] = l_bev_raw.detach()
                lam_bev = self._qal_weights(da_bev_logits, domains)
                l_bev = (lam_bev * l_bev_raw) if (lam_bev is not None) else l_bev_raw
                loss_dict['loss_da_bev'] = (l_bev * gate) * w_bev
            else:
                loss_dict['loss_da_bev'] = all_cls_scores[-1].sum() * 0.0

        # ===== QST（分类项）=====
        if self.qst_enable:
            p_iv = preds_dicts.get('p_iv')   # (B,Q,C)
            p_bev = preds_dicts.get('p_bev') # (B,Qb,C)
            if p_iv is None:
                p_iv = safe_softmax(all_cls_scores[-1], dim=-1)[..., :self.num_classes]
            if p_bev is None:
                p_bev = p_iv

            p_bev_reduce = p_bev.mean(dim=1, keepdim=True).expand(-1, p_iv.shape[1], -1)  # (B,Q,C)
            p_hat_cls = torch.maximum(p_bev_reduce, p_iv)
            p_hat_det = p_bev_reduce * p_iv

            p_mean = p_hat_det.mean(dim=1).mean(dim=0)  # (C,)
            self._qst_update_queue(p_mean)

            tau = self._qst_thresholds()  # (C,)
            with torch.no_grad():
                mask_iv = p_hat_cls >= tau.view(1, 1, -1)
                pseudo_y = p_hat_cls.argmax(dim=-1)
                valid = mask_iv.gather(-1, pseudo_y.unsqueeze(-1)).squeeze(-1)

            logits_last = all_cls_scores[-1][..., :self.num_classes]
            if valid.any():
                logits_sel = logits_last[valid]
                target_sel = pseudo_y[valid]
                qst_cls_loss = F.cross_entropy(logits_sel, target_sel)
                loss_dict['loss_qst_cls'] = qst_cls_loss * self.qst_loss_w
            else:
                loss_dict['loss_qst_cls'] = logits_last.sum() * 0.0

            if self.qst_use_det_loss:
                loss_dict['loss_qst_det'] = loss_dict['loss_qst_cls'].detach() * 0.0

        # ===== 2D Image-View 多标签辅助损失 L^iv =====
        if self.use_aux_2d and iv_logits is not None:
            B, Qiv, C = iv_logits.shape
            iv_prob = torch.sigmoid(torch.nan_to_num(iv_logits))
            tau = self._qst_thresholds().to(iv_logits.device)  # (C,)

            targets = []
            for b in range(B):
                if domains[b] == 0:  # source: 用 GT 类别集合
                    if len(gt_labels_list[b]) == 0:
                        y = torch.zeros(C, device=iv_logits.device)
                    else:
                        cls_idx = gt_labels_list[b].unique()
                        y = torch.zeros(C, device=iv_logits.device)
                        y[cls_idx] = 1.0
                else:                 # target: 用阈值化的弱标签
                    p_mean = iv_prob[b].mean(dim=0)   # (C,)
                    y = (p_mean >= tau).float()
                targets.append(y)
            targets = torch.stack(targets, dim=0)             # (B,C)
            targets = targets.unsqueeze(1).repeat(1, Qiv, 1)  # (B,Qiv,C)

            if torch.isfinite(iv_logits).all() and torch.isfinite(targets).all():
                loss_iv = self.loss_iv(iv_logits, targets)
                aux_w = float((self.da_cfg.get('loss_weights', {}) or {}).get('aux2d', 1.0))
                loss_dict['loss_iv_aux'] = loss_iv * aux_w
            else:
                loss_dict['loss_iv_aux'] = all_cls_scores[-1].sum() * 0.0

            # （可选）BEV 熵正则
            lw = (self.da_cfg.get('loss_weights', {}) if isinstance(self.da_cfg, dict) else {})
            w_bev_ent = float(lw.get('bev_ent', 0.0))
            if w_bev_ent > 0:
                bev_cls_scores = preds_dicts.get('bev_cls_scores', None)
                if bev_cls_scores is not None:
                    logp = F.log_softmax(bev_cls_scores[-1][..., :self.num_classes], dim=-1)  # (B,Q,C)
                    p = logp.exp()
                    ent = -(p * logp).sum(-1).mean()
                    loss_dict['loss_bev_ent'] = ent * w_bev_ent

        # 极小稳定项
        logits_last = preds_dicts['all_cls_scores'][-1]
        loss_dict['loss_stab'] = (logits_last.float().pow(2).mean()) * 1e-6

        # 最后统一把 NaN/Inf 清掉
        for k in list(loss_dict.keys()):
            loss_dict[k] = torch.nan_to_num(loss_dict[k])

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        ret_list = []
        for i in range(len(preds_dicts)):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            # 把 z 中心转回底部中心
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # 更健壮：优先从 img_metas 取 box_type_3d，否则兜底
            box_type = img_metas[i].get('box_type_3d', None)
            if box_type is None:
                mode = img_metas[i].get('box_mode_3d', 'LiDAR')
                if isinstance(mode, str):
                    m = mode.lower()
                    if 'depth' in m:
                        box_type = DepthInstance3DBoxes
                    elif 'camera' in m:
                        box_type = CameraInstance3DBoxes
                    else:
                        box_type = LiDARInstance3DBoxes
                else:
                    box_type = LiDARInstance3DBoxes

            bboxes = box_type(bboxes, bboxes.size(-1))
            ret_list.append([bboxes, preds['scores'], preds['labels']])
        return ret_list

# 新增一个 Hook（放到工程里任意 hooks.py）
from mmcv.runner import HOOKS, Hook
import torch

@HOOKS.register_module()
class ClipDAHook(Hook):
    def __init__(self, max_norm_disc=0.5):
        self.max_norm_disc = float(max_norm_disc)

    def after_backward(self, runner):
        disc_params = []
        for n, p in runner.model.named_parameters():
            if p.grad is None:
                continue
            if 'img_da_disc' in n or 'bev_da_disc' in n:
                disc_params.append(p)
        if disc_params:
            torch.nn.utils.clip_grad_norm_(disc_params, self.max_norm_disc)

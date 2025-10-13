# projects/mmdet3d_plugin/hooks/da_qst_monitor.py
from mmcv.runner import Hook
from mmcv.runner.hooks import HOOKS
import torch
import torch.nn.functional as F

@HOOKS.register_module()
class DAQSTMonitorHook(Hook):
    def after_train_iter(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        head = getattr(model, 'pts_bbox_head', None)
        if head is None or not hasattr(head, 'da_cfg'):
            return
        log_vars = {}

        # GRL λ
        if hasattr(head, 'grl'):
            try:
                log_vars['grl_lambda'] = float(head.grl._lambda)
            except Exception:
                pass

        # Access last forward outputs cached on runner (需你在训练循环把 preds_dicts 暂存到 runner)
        preds = getattr(runner, 'last_preds', None)
        metas = getattr(runner, 'last_metas', None)
        if preds is None or metas is None:
            return

        domains = preds.get('domains', None)
        if domains is not None:
            # acc_da_img
            if preds.get('da_img_logits', None) is not None:
                acc = (preds['da_img_logits'].argmax(dim=-1) == domains).float().mean()
                log_vars['acc_da_img'] = float(acc.detach().cpu())
            # acc_da_bev
            if preds.get('da_bev_logits', None) is not None:
                acc = (preds['da_bev_logits'].argmax(dim=-1) == domains).float().mean()
                log_vars['acc_da_bev'] = float(acc.detach().cpu())

        # QST 阈值统计
        if getattr(head, 'qst_enable', False):
            tau = head._qst_thresholds().detach().cpu()
            log_vars['tau_mean'] = float(tau.mean())
            log_vars['tau_min'] = float(tau.min())
            log_vars['tau_max'] = float(tau.max())

            # 伪标签命中率（上一 iter）
            p_iv = preds.get('p_iv', None)
            if p_iv is not None:
                tau = tau.to(p_iv.device).view(1, 1, -1)
                p_bev = preds.get('p_bev', p_iv).mean(dim=1, keepdim=True).expand_as(p_iv)
                p_hat = torch.maximum(p_bev, p_iv)
                pseudo_y = p_hat.argmax(dim=-1, keepdim=True)        # (B,Q,1)
                valid = p_hat.gather(-1, pseudo_y).squeeze(-1) >= tau # (B,Q)
                log_vars['qst_valid_ratio'] = float(valid.float().mean().detach().cpu())

        # 写日志
        runner.log_buffer.update(log_vars, runner.outputs['num_samples'])

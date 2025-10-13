from mmcv.runner import Hook
from mmcv.runner.hooks import HOOKS

@HOOKS.register_module()
class FirstBatchProbeHook(Hook):
    """首个 iter 打印图像/相机数/域分布，方便排查管线。"""
    def __init__(self):
        self._done = False

    def after_train_iter(self, runner):
        if self._done:
            return
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        # 期望 detector 在 forward_train 缓存了 last_metas（可选）
        metas = getattr(model, 'last_metas', None)
        imgsz = None
        if metas and isinstance(metas, list) and len(metas) > 0:
            # 统计 domain
            doms = [m.get('domain', 'unknown') for m in metas]
            cnt = {}
            for d in doms:
                cnt[d] = cnt.get(d, 0) + 1
            # 图像 shape（多相机）
            pad_shape = metas[0].get('pad_shape', None)
            if pad_shape:
                imgsz = pad_shape[0]  # (H,W,C) of one cam
            runner.logger.info(f'[FirstBatchProbe] domain_count={cnt}, image_size={imgsz}')
        else:
            runner.logger.info('[FirstBatchProbe] meta not found (this is fine)')
        self._done = True

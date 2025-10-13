from mmcv.runner import Hook
from mmcv.runner.hooks import HOOKS

@HOOKS.register_module()
class GRLSchedulerHook(Hook):
    def before_train_epoch(self, runner):
        # mmddp: .module
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        head = getattr(model, 'pts_bbox_head', None)
        if head is not None and hasattr(head, 'update_grl_lambda'):
            head.update_grl_lambda(runner.epoch)

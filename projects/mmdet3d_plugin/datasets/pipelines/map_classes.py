from mmdet.datasets import PIPELINES

@PIPELINES.register_module()
class MapClasses:
    def __init__(self, mapping, ignore_label='ignore'):
        self.mapping = mapping
        self.ignore_label = ignore_label

    def __call__(self, results):
        if 'gt_names' in results:
            results['gt_names'] = [
                self.mapping.get(n, self.ignore_label) for n in results['gt_names']
            ]
        return results

from neuro_pilot.engine.task import BaseTask
from neuro_pilot.core.registry import Registry
import torch.nn as nn
import torch
from neuro_pilot.nn.modules import Detect


@Registry.register_task("detect")
class DetectionTask(BaseTask):
    def build_model(self) -> nn.Module:
        if self.backbone:
            nc = self.cfg.head.num_classes
            det_head = Detect(nc=nc, ch=(128, 128, 128))
            det_head.stride = torch.tensor([8.0, 16.0, 32.0])
            self.model = det_head
            return det_head
        raise NotImplementedError

    def build_criterion(self) -> nn.Module:
        from neuro_pilot.utils.losses import DetectionLossAtomic

        return DetectionLossAtomic(self.backbone, self.model, self.cfg)

    def get_validator(self):
        from neuro_pilot.utils.metrics import DetectionMetric

        return DetectionMetric(self.cfg, "cuda", self.model)

    def get_trainer(self):
        """
        Return the trainer for this task.
        For atomic tasks used in composition, this might not be used directly
        if the CompositeTrainer handles the loop.
        """
        return None

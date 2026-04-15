import torch
from pathlib import Path

class BaseValidator:
    """
    Standardized Base Validator for NeuroPilot.
    Unifies metrics computation and evaluation logic.
    """
    def __init__(self, cfg, model, criterion, device):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.device = device
        self.log_dir = Path("experiments") / cfg.trainer.experiment_name / "val"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {}
        self.evaluator = None
        from .callbacks import CallbackList
        self.callbacks = CallbackList()

    def __call__(self, dataloader):
        """evaluation entry point."""
        self.model.eval()
        self.init_metrics()
        self.callbacks.on_val_start(self)

        with torch.no_grad():
            self.run_val_loop(dataloader)

        self.callbacks.on_val_end(self)
        return self.compute_final_metrics()

    def init_metrics(self):
        """Initialize metrics and evaluators."""
        raise NotImplementedError

    def run_val_loop(self, dataloader):
        """Logic for iterating over the validation set."""
        raise NotImplementedError

    def compute_final_metrics(self):
        """Final metrics computation and logging."""
        raise NotImplementedError

    def postprocess(self, preds):
        """Apply NMS or other post-processing."""
        return preds

from neuro_pilot.utils.metrics import DetectionEvaluator, TrajectoryMetric

class Validator(BaseValidator):
    """
    MultiTask Validator.
    Task-aware: only runs metrics/evaluation for active tasks.
    """
    def __init__(self, config, model, criterion, device, active_tasks=None):
        super().__init__(config, model, criterion, device)
        self.active_tasks = active_tasks or {'trajectory', 'detection', 'heatmap', 'gate'}

        # Only init detection decoder if detection is active
        if 'detection' in self.active_tasks:
            from neuro_pilot.utils.losses import DetectionLoss
            self.decoder = DetectionLoss(model) if 'detect' in getattr(model, 'heads', {}) else None
        else:
            self.decoder = None

        from neuro_pilot.utils.ops import get_bathtub_weights
        self.get_bathtub_weights = get_bathtub_weights

    def init_metrics(self):
        """Initialize metrics and evaluators — only for active tasks."""
        # Detection evaluator: only when detection task is active
        if 'detection' in self.active_tasks:
            names = getattr(self, 'names', getattr(self.model, 'names', None))
            self.evaluator = DetectionEvaluator(self.cfg.head.num_classes, self.device, self.log_dir, names=names)
        else:
            self.evaluator = None

        # Trajectory metrics: always active (ADE, FDE, lateral, longitudinal, L1, wL1)
        self.traj_metric = TrajectoryMetric(
            tau_start=self.cfg.loss.fdat_tau_start,
            tau_end=self.cfg.loss.fdat_tau_end
        )
        self.total_loss = 0.0

    def run_val_loop(self, dataloader):
        """Task-aware validation loop. Only runs detection eval when detection is active."""
        from neuro_pilot.utils.tqdm import TQDM
        pbar = TQDM(dataloader, desc="Validating")

        for i, batch in enumerate(pbar):
            self.batch_idx = i
            self.callbacks.on_val_batch_start(self)

            from neuro_pilot.utils.torch_utils import prepare_batch
            batch = prepare_batch(batch, self.device)
            img = batch['image']
            gt_wp = batch['waypoints']
            gt_boxes = batch['bboxes']
            gt_classes = batch.get('cls', batch.get('categories'))

            with torch.amp.autocast('cuda', enabled=True):
                model_kwargs = {k: v for k, v in batch.items() if k not in ('image', 'targets', 'image_path')}
                if 'clip_images' in batch and batch['clip_images'] is not None:
                     batch['clip_images'] = batch['clip_images'].to(self.device, non_blocking=True)
                     model_kwargs['clip_images'] = batch['clip_images']
                preds = self.model(img, **model_kwargs)
            self.current_output = preds

            targets = {
                'waypoints': gt_wp,
                'waypoints_mask': batch.get('waypoints_mask', torch.ones(img.size(0), device=self.device)),
                'bboxes': gt_boxes,
                'cls': gt_classes,
                'batch_idx': batch.get('batch_idx', torch.zeros(0, device=self.device)),
                'curvature': batch.get('curvature', torch.zeros(img.size(0), device=self.device)),
                'command_idx': batch.get('command_idx', torch.zeros(img.size(0), dtype=torch.long, device=self.device)),
                'action_target': batch.get('action_target', None)
            }
            batch['targets'] = targets
            self.current_batch = batch

            with torch.amp.autocast('cuda', enabled=True):
                loss_dict = self.criterion(preds, targets)
            loss_val = loss_dict['total']
            if torch.isfinite(loss_val):
                self.total_loss += loss_val.item()

            # --- Trajectory metrics (always active — ADE, FDE, lateral, longitudinal) ---
            pred_path = preds.get('waypoints', preds.get('control_points'))
            if pred_path is not None:
                traj_batch = {'waypoints': gt_wp}
                traj_preds = {'waypoints': pred_path}
                self.traj_metric.update(traj_preds, traj_batch)

            # --- Detection metrics (only when detection is active) ---
            if 'detection' in self.active_tasks and self.evaluator is not None:
                if 'bboxes' not in preds:
                    self.callbacks.on_val_batch_end(self)
                    continue

                bboxes = preds['bboxes']
                pred_bboxes = bboxes[:, :4, :].permute(0, 2, 1)
                pred_scores = bboxes[:, 4:, :].permute(0, 2, 1)

                formatted_preds = []
                formatted_targets = []

                batch_idx = batch.get('batch_idx', torch.zeros(gt_boxes.shape[0], device=self.device)).view(-1)

                for j in range(img.size(0)):
                    scores, labels = pred_scores[j].max(dim=1)
                    mask = scores > 0.001
                    k_boxes = pred_bboxes[j][mask]
                    k_scores = scores[mask]
                    k_labels = labels[mask]

                    if k_boxes.numel() > 0:
                        from torchvision.ops import nms
                        x1 = k_boxes[:, 0] - k_boxes[:, 2]/2
                        y1 = k_boxes[:, 1] - k_boxes[:, 3]/2
                        x2 = k_boxes[:, 0] + k_boxes[:, 2]/2
                        y2 = k_boxes[:, 1] + k_boxes[:, 3]/2
                        xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                        keep = nms(xyxy, k_scores, 0.6)
                        formatted_preds.append({'boxes': xyxy[keep], 'scores': k_scores[keep], 'labels': k_labels[keep]})
                    else:
                        formatted_preds.append({'boxes': torch.empty((0, 4), device=self.device), 'scores': torch.tensor([], device=self.device), 'labels': torch.tensor([], device=self.device)})

                    mask_j = (batch_idx == j)
                    t_boxes = gt_boxes[mask_j]
                    t_labels = gt_classes[mask_j]

                    if t_boxes.numel() > 0:
                        h, w_img = img.shape[2], img.shape[3]
                        tx1 = (t_boxes[:, 0] - t_boxes[:, 2]/2) * w_img
                        ty1 = (t_boxes[:, 1] - t_boxes[:, 3]/2) * h
                        tx2 = (t_boxes[:, 0] + t_boxes[:, 2]/2) * w_img
                        ty2 = (t_boxes[:, 1] + t_boxes[:, 3]/2) * h
                        t_xyxy = torch.stack([tx1, ty1, tx2, ty2], dim=1)
                        formatted_targets.append({'boxes': t_xyxy, 'labels': t_labels.view(-1)})
                    else:
                        formatted_targets.append({'boxes': torch.empty((0, 4), device=self.device), 'labels': torch.tensor([], device=self.device)})

                self.evaluator.update(formatted_preds, formatted_targets)

            self.callbacks.on_val_batch_end(self)

    def compute_final_metrics(self):
        """Task-aware final metrics computation."""
        metric_res = {}

        # Detection metrics — only when active
        if 'detection' in self.active_tasks and self.evaluator is not None:
            det_metrics = self.evaluator.compute()
            metric_res.update(det_metrics)
            self.evaluator.plot_confusion_matrix()

        # Trajectory metrics — always (L1, wL1, ADE, FDE, lateral, longitudinal)
        traj_results = self.traj_metric.compute()
        metric_res.update(traj_results)
        metric_res['avg_loss'] = self.total_loss / max(1, self.batch_idx + 1)

        # Ensure names are passed to trainer for summary
        if hasattr(self, 'names') and self.names:
            metric_res['names'] = self.names

        return metric_res

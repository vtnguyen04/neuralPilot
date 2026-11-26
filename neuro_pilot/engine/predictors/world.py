"""World Model predictors (CommaVQ, LatentDrive V-JEPA, CommaGPT, CommaDiT, ViT-AE)."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence, Union

import cv2
import numpy as np
import torch

from neuro_pilot.engine.predictors.base import PREDICTOR_REGISTRY, BaseGraphPredictor
from neuro_pilot.engine.predictors.utils import crop_2to1, load_bgr, to_imagenet_tensor
from neuro_pilot.nn.zoo.blocks.world import sample_dit_euler
from neuro_pilot.nn.zoo.graph import GraphModel


@PREDICTOR_REGISTRY.register("vq_enc")
@PREDICTOR_REGISTRY.register("comma_vq_encoder")
@PREDICTOR_REGISTRY.register("commavq_enc")
class VqEncPredictor(BaseGraphPredictor):
    """Encode BGR road images into `8x16` discrete VQGAN token indices."""

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            return super().preprocess(source)
        shape = self.model.input_specs["big_img"]["shape"]
        img_h, img_w = int(shape[2]), int(shape[3])
        rgb = cv2.cvtColor(crop_2to1(load_bgr(source), img_w, img_h), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.astype(np.float32).transpose(2, 0, 1)[None])
        ).to(self.device)
        return {"big_img": tensor}

    def postprocess(
        self, raw_outputs: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, Any]:
        tokens = raw_outputs["encoding_indices"].detach().cpu()
        return {"tokens": tokens, "tokens_np": tokens.numpy()}


@PREDICTOR_REGISTRY.register("vq_dec")
@PREDICTOR_REGISTRY.register("comma_vq_decoder")
@PREDICTOR_REGISTRY.register("commavq_dec")
class VqDecPredictor(BaseGraphPredictor):
    """Decode `8x16` discrete VQGAN token indices back into BGR road frames."""

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            return super().preprocess(source)
        tokens = torch.as_tensor(source, dtype=torch.int64, device=self.device)
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        return {"encoding_indices": tokens}

    def postprocess(
        self, raw_outputs: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, Any]:
        raw = raw_outputs["big_decoded_img"].detach().cpu().numpy()[0].transpose(1, 2, 0)
        rgb = np.clip((raw + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        return {"image_bgr": cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "raw": raw}


@PREDICTOR_REGISTRY.register("jepa")
@PREDICTOR_REGISTRY.register("latent_drive_vjepa")
@PREDICTOR_REGISTRY.register("latent_drive")
class JepaPredictor(BaseGraphPredictor):
    """4-frame V-JEPA 3D-RoPE world-model trajectory planner predictor."""

    def __init__(
        self,
        model: GraphModel,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(model, device=device)
        self._clip_buf: Deque[torch.Tensor] = deque(maxlen=4)

    def reset_state(self) -> None:
        self._clip_buf.clear()

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            return super().preprocess(source)
        shape = self.model.input_specs["video"]["shape"]
        img_h, img_w = int(shape[3]), int(shape[4])
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            frames = [to_imagenet_tensor(load_bgr(item), img_w, img_h)[0] for item in source]
            while len(frames) < 4:
                frames.insert(0, frames[0])
            clip = torch.stack(frames[-4:], dim=1).unsqueeze(0).to(self.device)
        else:
            frame_t = to_imagenet_tensor(load_bgr(source), img_w, img_h)[0]
            self._clip_buf.append(frame_t)
            frames = list(self._clip_buf)
            while len(frames) < 4:
                frames.insert(0, frames[0])
            clip = torch.stack(frames, dim=1).unsqueeze(0).to(self.device)

        status_val = kwargs.get("status")
        status = (
            torch.as_tensor(status_val, dtype=torch.float32, device=self.device).reshape(1, 16)
            if status_val is not None
            else torch.zeros((1, 16), dtype=torch.float32, device=self.device)
        )
        return {"video": clip, "status": status}

    def postprocess(
        self, raw_outputs: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, Any]:
        plan = raw_outputs["plan"].detach().cpu().numpy()[0]
        return {"plan": plan}


@PREDICTOR_REGISTRY.register("gpt")
@PREDICTOR_REGISTRY.register("comma_gpt2m_world_model")
@PREDICTOR_REGISTRY.register("comma_gpt")
class GptPredictor(BaseGraphPredictor):
    """Autoregressive discrete token predictor for CommaGPT."""

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            return super().preprocess(source)
        ids = torch.as_tensor(source, dtype=torch.int64, device=self.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        return {"input_ids": ids}


@PREDICTOR_REGISTRY.register("dit")
@PREDICTOR_REGISTRY.register("comma_worldmodel_dit")
@PREDICTOR_REGISTRY.register("comma_dit")
class DitPredictor(BaseGraphPredictor):
    """Rectified-flow video DiT predictor with optional multi-step Euler sampling."""

    def predict(self, source: Any, **kwargs: Any) -> Dict[str, Any]:
        steps = kwargs.get("sample_steps")
        if steps is not None and isinstance(source, dict):
            feed = self.preprocess(source)
            ctx_lat = feed["latents"]
            if ctx_lat.size(1) >= feed["rel_pose"].size(1):
                ctx_lat = ctx_lat[:, : feed["rel_pose"].size(1) - 1]
            sampled = sample_dit_euler(
                self.model,
                ctx_lat,
                feed["rel_pose"],
                feed["pose_mask"],
                feed["frame_indices"],
                num_steps=int(steps),
            )
            return {k: v.detach().cpu().numpy() for k, v in sampled.items()}
        return super().predict(source, **kwargs)


@PREDICTOR_REGISTRY.register("ae")
@PREDICTOR_REGISTRY.register("comma_vit_ae_f8c32")
@PREDICTOR_REGISTRY.register("comma_ae")
class AePredictor(BaseGraphPredictor):
    """Paired dual-camera ViT autoencoder predictor."""

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            return super().preprocess(source)
        in_name, spec = next(iter(self.model.input_specs.items()))
        shape = spec["shape"] if isinstance(spec, dict) else spec
        img_h, img_w = int(shape[2]), int(shape[3])
        narrow = to_imagenet_tensor(load_bgr(source), img_w, img_h)
        wide_src = kwargs.get("wide_img")
        wide = to_imagenet_tensor(load_bgr(wide_src), img_w, img_h) if wide_src else narrow
        return {in_name: torch.cat((narrow, wide), dim=1).to(self.device)}

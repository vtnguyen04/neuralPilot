"""Generative world model blocks for CommaGPT and CommaDiT."""

from __future__ import annotations

import math
from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY
from neuro_pilot.nn.zoo.blocks.common import MLP, RMSNorm

AnyState = Union[torch.Tensor, Dict[str, Any]]


@BLOCK_REGISTRY.register("TokenPosEmbed")
class TokenPosEmbed(nn.Module):
    """Token and position embeddings with optional KV-cache state setup."""

    def __init__(
        self,
        vocab_size: int = 1025,
        n_positions: int = 2580,
        dim: int = 1024,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_positions = n_positions
        self.wte = nn.Embedding(vocab_size, dim)
        self.wpe = nn.Embedding(n_positions, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, inputs: AnyState) -> Dict[str, Any]:
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            past_kv = inputs.get("past_key_values", inputs.get("past_kv"))
            use_cache = bool(inputs.get("use_cache", False))
        else:
            input_ids = inputs
            past_kv = None
            use_cache = False

        seq_len = input_ids.size(1)
        past_len = past_kv[0][0].size(2) if past_kv else 0
        if past_len + seq_len > self.n_positions:
            raise ValueError(f"Sequence length {past_len + seq_len} exceeds {self.n_positions}")

        pos_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        hidden = self.drop(self.wte(input_ids) + self.wpe(pos_ids).unsqueeze(0))
        return {
            "x": hidden,
            "past_kv": past_kv,
            "present_kv": [] if use_cache else None,
            "use_cache": use_cache,
            "layer_idx": 0,
        }


@BLOCK_REGISTRY.register("GPTOut")
class GPTOut(nn.Module):
    """Final LayerNorm and LM head returning logits or `(logits, present_kv)`."""

    def __init__(self, dim: int = 1024, vocab_size: int = 1025, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, state: AnyState) -> Any:
        hidden = state["x"] if isinstance(state, dict) else state
        logits = self.lm_head(self.norm(hidden))
        if isinstance(state, dict) and state.get("use_cache", False):
            return logits, state["present_kv"]
        return logits


def _sinusoidal_time_emb(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    steps = torch.arange(half, device=timesteps.device, dtype=torch.float32)
    freqs = torch.exp(-math.log(10000.0) * steps / max(half, 1))
    angles = timesteps.float().unsqueeze(-1) * freqs
    emb = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(emb, (0, dim - emb.shape[-1]))


@BLOCK_REGISTRY.register("DiTPatchIn")
class DiTPatchIn(nn.Module):
    """Patchify video latents and build frame conditioning and block-causal mask."""

    unpack_inputs = True

    def __init__(
        self,
        latent_ch: int = 32,
        latent_h: int = 16,
        latent_w: int = 32,
        patch_size: int = 2,
        dim: int = 256,
        cond_dim: int = 256,
        pose_dim: int = 6,
        max_frames: int = 256,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.cond_dim = cond_dim
        self.max_frames = max_frames
        self.tokens_per_frame = (latent_h // patch_size) * (latent_w // patch_size)
        self.patch_in = nn.Linear(latent_ch * patch_size * patch_size, dim)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, self.tokens_per_frame, dim) * 0.02)
        self.frame_embed = nn.Embedding(max_frames, cond_dim)
        self.pose_proj = nn.Linear(pose_dim + 1, cond_dim)
        self.time_mlp = MLP(cond_dim, cond_dim, out_dim=cond_dim, act="silu")

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rel_pose: torch.Tensor,
        pose_mask: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, frames, channels, height, width = latents.shape
        patch = self.patch_size
        grid = latents.reshape(
            batch, frames, channels, height // patch, patch, width // patch, patch
        )
        flat = grid.permute(0, 1, 3, 5, 2, 4, 6).reshape(
            batch, frames, self.tokens_per_frame, channels * patch * patch
        )
        tokens = self.patch_in(flat) + self.spatial_pos

        time_emb = self.time_mlp(_sinusoidal_time_emb(timesteps, self.cond_dim))
        pose_in = torch.cat((rel_pose * pose_mask.unsqueeze(-1), pose_mask.unsqueeze(-1)), dim=-1)
        frame_emb = self.frame_embed(frame_indices.clamp(0, self.max_frames - 1))
        cond = time_emb + self.pose_proj(pose_in) + frame_emb

        causal = torch.tril(torch.ones((frames, frames), device=latents.device, dtype=torch.bool))
        mask = causal.repeat_interleave(self.tokens_per_frame, dim=0)
        mask = mask.repeat_interleave(self.tokens_per_frame, dim=1)
        return {"x": tokens, "cond": cond, "mask": mask}


@BLOCK_REGISTRY.register("DiTUnpatch")
class DiTUnpatch(nn.Module):
    """Apply RMSNorm and unpatchify DiT tokens back to video latent velocity."""

    def __init__(
        self,
        dim: int = 256,
        latent_ch: int = 32,
        latent_h: int = 16,
        latent_w: int = 32,
        patch_size: int = 2,
    ) -> None:
        super().__init__()
        self.latent_ch = latent_ch
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.patch_size = patch_size
        self.grid_h = latent_h // patch_size
        self.grid_w = latent_w // patch_size
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(dim, latent_ch * patch_size * patch_size)

    def forward(self, state: AnyState) -> torch.Tensor:
        tokens = state["x"] if isinstance(state, dict) else state
        batch, frames = tokens.shape[:2]
        patch = self.patch_size
        out = self.proj(self.norm(tokens))
        out = out.reshape(batch, frames, self.grid_h, self.grid_w, self.latent_ch, patch, patch)
        return out.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch, frames, self.latent_ch, self.latent_h, self.latent_w
        )


@BLOCK_REGISTRY.register("PlanHead")
class PlanHead(nn.Module):
    """Predict ego trajectory plan from the target frame DiT tokens."""

    def __init__(self, dim: int = 256, plan_horizon: int = 33, plan_dim: int = 3) -> None:
        super().__init__()
        self.plan_horizon = plan_horizon
        self.plan_dim = plan_dim
        self.norm = RMSNorm(dim)
        self.mlp = MLP(dim, dim, out_dim=plan_horizon * plan_dim, act="silu")

    def forward(self, state: AnyState) -> torch.Tensor:
        tokens = state["x"] if isinstance(state, dict) else state
        target = tokens[:, -1] if tokens.ndim == 4 else tokens
        pooled = self.norm(target.mean(dim=1))
        return self.mlp(pooled).reshape(pooled.size(0), self.plan_horizon, self.plan_dim)


@torch.no_grad()
def sample_dit_euler(
    model: nn.Module,
    context_latents: torch.Tensor,
    rel_pose: torch.Tensor,
    pose_mask: torch.Tensor,
    frame_indices: torch.Tensor,
    num_steps: int = 3,
) -> Dict[str, torch.Tensor]:
    """Run rectified-flow Euler sampling for one next frame on a DiT GraphModel."""
    batch, num_ctx, channels, height, width = context_latents.shape
    device, dtype = context_latents.device, context_latents.dtype
    x_t = torch.randn((batch, 1, channels, height, width), device=device, dtype=dtype)
    t_ctx = torch.zeros((batch, num_ctx), device=device, dtype=dtype)
    dt = 1.0 / max(num_steps, 1)
    plan = torch.zeros((batch, 1, 3), device=device, dtype=dtype)

    for step in range(num_steps):
        t_cur = torch.full((batch, 1), 1.0 - step * dt, device=device, dtype=dtype)
        feed = {
            "latents": torch.cat((context_latents, x_t), dim=1),
            "timesteps": torch.cat((t_ctx, t_cur), dim=1),
            "rel_pose": rel_pose[:, : num_ctx + 1],
            "pose_mask": pose_mask[:, : num_ctx + 1],
            "frame_indices": frame_indices[:, : num_ctx + 1],
        }
        out = model.forward_onnx_dict(feed)
        x_t = x_t - dt * out["velocity"][:, -1:]
        plan = out["plan"]
    return {"latent": x_t, "plan": plan}

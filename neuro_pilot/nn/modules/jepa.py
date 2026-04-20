import torch
import torch.nn as nn
from .mlp import MLP
from .regularization import SIGReg


class JEPAPredictor(nn.Module):
    """
    Joint-Embedding Predictive Architecture (JEPA) Predictor.
    Integrates the pristine SIGReg and MLP modules from le-wm.
    """

    def __init__(self, in_channels, hidden_dim=None, num_knots=17, num_proj=1024):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_channels * 2

        self.predictor = MLP(
            input_dim=in_channels,
            hidden_dim=hidden_dim,
            output_dim=in_channels,
            norm_fn=nn.BatchNorm1d,
            act_fn=nn.GELU,
        )

        self.sigreg = SIGReg(knots=num_knots, num_proj=num_proj)

    def forward(self, x):
        """
        x: Context latent feature from backbone P5. Shape: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Flatten spatial dimensions for JEPAPredictor MLP: (B*H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)

        # Calculate sketch isotropic gaussian regularizer on contextual embedding
        # SIGReg is designed for image-level embeddings (like ViT CLS token).
        # We apply Global Average Pooling to prevent OOM and accurately represent the scene.
        x_pooled = x.mean(dim=[2, 3])  # (B, C)
        sigreg_input = x_pooled.unsqueeze(0)  # (1, B, C)
        l_sigreg = self.sigreg(sigreg_input)

        # Predict target latent
        pred_flat = self.predictor(x_flat)
        pred = pred_flat.view(B, H, W, C).permute(0, 3, 1, 2)

        return {"jepa_pred": pred, "jepa_context": x, "sigreg_loss": l_sigreg}

"""
DnCNN model for blind Gaussian denoising of coronary angiography frames.

Implements residual learning: the network predicts the noise component and
subtracts it from the input to recover the clean image (Zhang et al. 2017,
"Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image
Denoising", IEEE TIP).

Two operating modes
-------------------

single
    Input ``(B, 1, H, W)`` — one noisy frame.
    Output ``(B, 1, H, W)`` — denoised frame.

temporal
    Input ``(B, 3, H, W)`` — noisy stack (t-1, t, t+1) from TemporalDataset.
    Output ``(B, 1, H, W)`` — denoised centre frame (t).
    The residual is subtracted from channel index 1 (the centre frame).

Use ``build_dncnn(mode)`` to construct the appropriate variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# DnCNN
# ---------------------------------------------------------------------------

class DnCNN(nn.Module):
    """Residual-learning denoising CNN (Zhang et al. 2017).

    Architecture:
      - Layer 1:  Conv(in_channels → features, 3×3) + ReLU
      - Layers 2…depth-1: Conv(features → features, 3×3) + BN + ReLU
      - Layer depth: Conv(features → out_channels, 3×3)
      - Output: input_clean_channels − noise_estimate

    For temporal mode (in_channels == 3) "input_clean_channels" is channel 1
    (the centre frame) so the output shape is always (B, out_channels, H, W).

    Args:
        in_channels:  Number of input channels (1 for single, 3 for temporal).
        out_channels: Number of output channels (always 1 — one clean frame).
        depth:        Total number of conv layers (17 per Zhang et al. 2017).
        features:     Number of feature maps in hidden layers (64).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 17,
        features: int = 64,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be ≥ 2, got {depth}")

        layers: list[nn.Module] = []

        # First layer — no BN
        layers += [
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]

        # Middle layers — Conv + BN + ReLU
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ]

        # Last layer — no BN, no activation
        layers.append(
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=True),
        )

        self.net = nn.Sequential(*layers)
        self._in_channels = in_channels
        self._weight_init()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _weight_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            # BN layers keep PyTorch defaults (weight=1, bias=0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the clean image via residual subtraction.

        Args:
            x: Input tensor ``(B, in_channels, H, W)``.

        Returns:
            Denoised tensor ``(B, out_channels, H, W)``.
        """
        noise = self.net(x)

        # For temporal input (3 channels) subtract from centre frame only.
        if self._in_channels == 3:
            clean_ref = x[:, 1:2, :, :]    # (B, 1, H, W) — channel t
        else:
            clean_ref = x                   # (B, 1, H, W)

        return clean_ref - noise


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dncnn(mode: str, depth: int = 17) -> DnCNN:
    """Construct a DnCNN for the given operating mode.

    Args:
        mode:  ``"single"`` (1 input channel) or ``"temporal"`` (3 input channels).
        depth: Number of convolutional layers (17 per Zhang et al. 2017).

    Returns:
        Configured :class:`DnCNN` instance.

    Raises:
        ValueError: If *mode* is not ``"single"`` or ``"temporal"``.
    """
    if mode == "single":
        return DnCNN(in_channels=1, out_channels=1, depth=depth)
    elif mode == "temporal":
        return DnCNN(in_channels=3, out_channels=1, depth=depth)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'single' or 'temporal'.")


# ---------------------------------------------------------------------------
# Quick self-test (run with:  python -m models.dncnn)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(0)

    def _count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    # Build both variants
    single_model   = build_dncnn("single")
    temporal_model = build_dncnn("temporal")

    print(f"DnCNN (single)   parameters: {_count_params(single_model):,}")
    print(f"DnCNN (temporal) parameters: {_count_params(temporal_model):,}")

    # Forward pass — shapes match our dataset outputs
    single_input  = torch.randn(2, 1, 128, 128)
    single_target = torch.randn(2, 1, 128, 128)

    temporal_input  = torch.randn(2, 3, 128, 128)
    temporal_target = torch.randn(2, 1, 128, 128)

    # Single-frame check
    single_out = single_model(single_input)
    assert single_out.shape == single_target.shape, (
        f"single output {single_out.shape} != target {single_target.shape}"
    )
    loss_s = F.mse_loss(single_out, single_target)
    loss_s.backward()
    print(f"single  — output {tuple(single_out.shape)}  MSE loss {loss_s.item():.4f}  [OK]")

    # Temporal check
    temporal_out = temporal_model(temporal_input)
    assert temporal_out.shape == temporal_target.shape, (
        f"temporal output {temporal_out.shape} != target {temporal_target.shape}"
    )
    loss_t = F.mse_loss(temporal_out, temporal_target)
    loss_t.backward()
    print(f"temporal — output {tuple(temporal_out.shape)}  MSE loss {loss_t.item():.4f}  [OK]")

    print("All checks passed.")

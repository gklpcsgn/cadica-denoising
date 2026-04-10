"""
DnCNN – Denoising Convolutional Neural Network (residual-learning variant).

Reference
---------
Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017).
Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising.
IEEE Transactions on Image Processing, 26(7), 3142–3155.
https://doi.org/10.1109/TIP.2017.2662206

Architecture
------------
Input  →  Conv(64, 3×3) + ReLU
           ×N  Conv(64, 3×3) + BN + ReLU     (default N = 15)
        →  Conv(out_channels, 3×3)
Output  =  input[:, mid_ch, ...] − predicted_noise   (residual learning)

The ``in_channels`` parameter lets the same class serve both:
  - Single-frame denoising  (in_channels=1)
  - Temporal denoising      (in_channels=3, centre channel = index 1)

When in_channels > 1 the residual connection uses only the **centre input
channel** (index in_channels // 2), which corresponds to the current frame
being denoised in the temporal case.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """Residual-learning CNN for image denoising.

    Args:
        in_channels: Number of input channels.
                     Use 1 for single-frame, 3 for temporal (t-1, t, t+1).
        out_channels: Number of output channels (almost always 1: the
                      denoised centre frame).
        num_layers: Total depth, counting the first Conv+ReLU layer and the
                    final Conv layer. Interior layers are (Conv+BN+ReLU).
                    Default 17 matches Zhang et al. for blind Gaussian noise;
                    values between 15–20 are typical for X-ray denoising.
        features: Number of feature maps in each intermediate layer (64 in
                  the original paper).
        bias: Whether to use bias in convolutional layers. The original paper
              omits bias in BN layers but keeps it in the first/last convs.
              BN layers absorb any bias in intermediate layers, so we disable
              bias there following the original implementation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_layers: int = 17,
        features: int = 64,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 3:
            raise ValueError("num_layers must be ≥ 3 (first layer + ≥1 BN layer + last layer).")

        # Index of the input channel to use for the residual shortcut
        # For in_channels=1 this is 0; for 3 it is 1 (the current frame).
        self._residual_ch = in_channels // 2

        layers: list[nn.Module] = []

        # --- First layer: Conv + ReLU (no BN) ---
        layers.append(nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # --- Intermediate layers: Conv + BN + ReLU ---
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=bias)
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # --- Last layer: Conv (no BN, no activation) ---
        layers.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=True))

        self.body = nn.Sequential(*layers)
        self._initialize_weights()

    # ------------------------------------------------------------------
    # Weight initialisation (He normal for ReLU networks)
    # ------------------------------------------------------------------

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise input tensor.

        Args:
            x: Input tensor of shape (B, C, H, W) where C == in_channels.

        Returns:
            Denoised tensor of shape (B, out_channels, H, W).
            The network predicts residual noise and subtracts it from the
            centre input channel, so the output lives in approximately the
            same value range as the input.
        """
        noise = self.body(x)                        # predicted noise residual
        centre = x[:, self._residual_ch : self._residual_ch + 1, :, :]  # (B,1,H,W)
        return centre - noise                        # denoised output


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_dncnn(mode: str = "single", **kwargs) -> DnCNN:
    """Convenience factory.

    Args:
        mode: "single" → in_channels=1, "temporal" → in_channels=3.
        **kwargs: Forwarded to DnCNN (num_layers, features, …).

    Returns:
        Configured DnCNN instance.
    """
    in_channels = 1 if mode == "single" else 3
    return DnCNN(in_channels=in_channels, **kwargs)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for mode, C in [("single", 1), ("temporal", 3)]:
        model = build_dncnn(mode)
        x = torch.randn(2, C, 128, 128)
        y = model(x)
        assert y.shape == (2, 1, 128, 128), y.shape
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"{mode:8s}  in_channels={C}  "
            f"output={tuple(y.shape)}  "
            f"params={n_params:,}"
        )

"""
Fusion-ResNet for Non-Intrusive Load Monitoring (NILM)

An advanced multi-branch architecture that fuses multiple feature representations
(raw signal, ICA-decomposed, Fryze power decomposition, FFT) through convolutional
ResNet blocks and an attention-based fusion module for multi-label appliance
classification.

Architecture:
    Input Signal (batch, signal_length=400)
        ├── Branch 1: Raw Signal → 1D Conv ResNet blocks
        ├── Branch 2: ICA decomposition → 1D Conv ResNet blocks
        ├── Branch 3: Fryze power decomposition → 1D Conv ResNet blocks
        └── Branch 4: FFT (frequency domain) → 1D Conv ResNet blocks

    All branches → Squeeze-and-Excitation Attention Fusion → Classification Head

Paper: Fusion-ResNet for NILM Energy Disaggregation
Base project: ICAResNetFFN (ML2023SK Team 37)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==============================================================================
# Building Blocks
# ==============================================================================

class ResBlock1D(nn.Module):
    """1D Convolutional Residual Block with pre-activation (BN → ReLU → Conv)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + residual


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1)
        return x * w


class ResStage(nn.Module):
    """A stage of stacked ResBlock1D blocks with optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        stride: int = 1,
        dropout: float = 0.1,
        use_se: bool = True,
    ):
        super().__init__()
        blocks = [ResBlock1D(in_channels, out_channels, stride=stride, dropout=dropout)]
        for _ in range(1, num_blocks):
            blocks.append(ResBlock1D(out_channels, out_channels, dropout=dropout))
        if use_se:
            blocks.append(SEBlock(out_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# ==============================================================================
# Preprocessing Layers (carried over from original project, made differentiable)
# ==============================================================================

class ICALayer(nn.Module):
    """Fixed ICA un-mixing layer (non-trainable)."""

    def __init__(self, U: np.ndarray, M: np.ndarray):
        super().__init__()
        self.register_buffer('U', torch.from_numpy(np.array(U, dtype=np.float64)))
        self.register_buffer('M', torch.from_numpy(np.array(M, dtype=np.float64)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.M) @ self.U.T


class NormalizeLayer(nn.Module):
    """Fixed normalization using pre-computed mean/std on exp-transformed ICA output."""

    def __init__(self, m: np.ndarray, s: np.ndarray):
        super().__init__()
        self.register_buffer('m', torch.from_numpy(np.array(m, dtype=np.float64)))
        self.register_buffer('s', torch.from_numpy(np.array(s, dtype=np.float64)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.exp(x)
        return (x - self.m) / self.s


class FryzeDecomposition(nn.Module):
    """Fryze power decomposition — extracts active and non-active currents.

    Assumes input signal has been reshaped to (batch, n_cycles, cycle_length),
    and a synthetic voltage waveform is generated internally.
    """

    def __init__(self, signal_length: int = 400, emb_size: int = 50):
        super().__init__()
        self.signal_length = signal_length
        self.emb_size = emb_size

        # Pre-compute voltage waveform (US 60Hz mains)
        n_cycles = signal_length // emb_size
        t = np.linspace(0, 1 / 60, emb_size)
        v = 120 * np.sqrt(2) * np.sin(2 * np.pi * 60 * t)
        self.register_buffer('voltage', torch.from_numpy(np.array(v, dtype=np.float64)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, signal_length)
        Returns:
            features: (batch, 2, emb_size) — [non_active; active] currents
        """
        batch = x.shape[0]
        # Reshape to cycles and average
        x = x.view(batch, -1, self.emb_size)  # (B, n_cycles, emb_size)
        x = x.mean(dim=1)  # (B, emb_size)

        v = self.voltage.unsqueeze(0).expand(batch, -1)  # (B, emb_size)

        # Fryze decomposition
        p = x * v  # instantaneous power
        v_rms_sq = (v ** 2).mean(dim=-1, keepdim=True)
        i_active = p.mean(dim=-1, keepdim=True) * v / v_rms_sq
        i_non_active = x - i_active

        # Stack: (batch, 2, emb_size)
        return torch.stack([i_non_active, i_active], dim=1)


# ==============================================================================
# Branch Networks
# ==============================================================================

class RawSignalBranch(nn.Module):
    """Branch 1: Processes the raw time-domain signal with 1D Conv ResNet."""

    def __init__(
        self,
        signal_length: int = 400,
        channels: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        stages = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            stages.append(ResStage(in_ch, out_ch, num_blocks=2, stride=2, dropout=dropout))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).squeeze(-1)
        return x


class ICABranch(nn.Module):
    """Branch 2: ICA decomposition → 1D Conv ResNet on component features."""

    def __init__(
        self,
        U: np.ndarray,
        M: np.ndarray,
        m: np.ndarray,
        s: np.ndarray,
        channels: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        self.ica = ICALayer(U, M)
        self.norm = NormalizeLayer(m, s)

        n_components = U.shape[0]

        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        stages = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            stages.append(ResStage(in_ch, out_ch, num_blocks=2, stride=1, dropout=dropout))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ica(x)   # (B, n_components)
        x = self.norm(x)
        x = x.unsqueeze(1)  # (B, 1, n_components) — treat components as sequence
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).squeeze(-1)
        return x


class FryzeBranch(nn.Module):
    """Branch 3: Fryze power decomposition → 1D Conv ResNet."""

    def __init__(
        self,
        signal_length: int = 400,
        emb_size: int = 50,
        channels: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        self.fryze = FryzeDecomposition(signal_length, emb_size)

        # Input has 2 channels (active, non-active)
        self.stem = nn.Sequential(
            nn.Conv1d(2, channels[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        stages = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            stages.append(ResStage(in_ch, out_ch, num_blocks=2, stride=2, dropout=dropout))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fryze(x)  # (B, 2, emb_size)
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).squeeze(-1)
        return x


class FFTBranch(nn.Module):
    """Branch 4: FFT frequency-domain features → 1D Conv ResNet."""

    def __init__(
        self,
        signal_length: int = 400,
        channels: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        # Only use first half of FFT (positive frequencies)
        self.n_freqs = signal_length // 2

        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        stages = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            stages.append(ResStage(in_ch, out_ch, num_blocks=2, stride=2, dropout=dropout))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute magnitude spectrum
        x_fft = torch.fft.rfft(x, dim=-1)
        x_mag = torch.abs(x_fft)[:, :self.n_freqs]
        x_mag = x_mag.unsqueeze(1)  # (B, 1, n_freqs)
        x_mag = self.stem(x_mag)
        x_mag = self.stages(x_mag)
        x_mag = self.pool(x_mag).squeeze(-1)
        return x_mag


# ==============================================================================
# Fusion Module
# ==============================================================================

class AttentionFusion(nn.Module):
    """Attention-weighted fusion of multi-branch features.

    Uses a learnable attention mechanism to weight the contribution
    of each branch before concatenation and projection.
    """

    def __init__(self, branch_dims: list[int], fused_dim: int, dropout: float = 0.1):
        super().__init__()
        total_dim = sum(branch_dims)
        self.n_branches = len(branch_dims)

        # Branch-level attention (gating)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(total_dim // 2, self.n_branches),
            nn.Softmax(dim=-1),
        )

        # Projection layers to equalize dimensions before weighted sum
        self.projections = nn.ModuleList([
            nn.Linear(d, fused_dim) for d in branch_dims
        ])

        self.norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, branch_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            branch_features: list of (batch, dim_i) tensors from each branch
        Returns:
            fused: (batch, fused_dim)
        """
        # Concatenate for gating
        concat = torch.cat(branch_features, dim=-1)  # (B, total_dim)
        attn_weights = self.gate(concat)  # (B, n_branches)

        # Project each branch to same dimension
        projected = [proj(feat) for proj, feat in zip(self.projections, branch_features)]
        projected = torch.stack(projected, dim=1)  # (B, n_branches, fused_dim)

        # Weighted sum
        attn_weights = attn_weights.unsqueeze(-1)  # (B, n_branches, 1)
        fused = (projected * attn_weights).sum(dim=1)  # (B, fused_dim)

        return self.dropout(self.norm(fused))


# ==============================================================================
# Main Model: Fusion-ResNet
# ==============================================================================

class FusionResNet(nn.Module):
    """
    Fusion-ResNet for Non-Intrusive Load Monitoring.

    Multi-branch architecture combining:
    - Raw signal convolutional path
    - ICA-decomposed feature path
    - Fryze power decomposition path
    - FFT frequency-domain path

    Features are fused via attention-weighted combination and classified
    through a multi-label head.

    Args:
        n_classes: Number of appliance classes for multi-label output.
        signal_length: Length of input signal (default: 400 for PLAID).
        U: ICA un-mixing matrix (n_components, signal_length).
        M: ICA mean vector (signal_length,).
        m: Mean for normalization.
        s: Std for normalization.
        branch_channels: Channel progression for each ResNet branch.
        fused_dim: Dimension of the fused feature vector.
        dropout: Dropout rate.
        emb_size: Embedding size for Fryze decomposition (cycle length).
    """

    def __init__(
        self,
        n_classes: int,
        signal_length: int = 400,
        U: np.ndarray = None,
        M: np.ndarray = None,
        m: np.ndarray = None,
        s: np.ndarray = None,
        branch_channels: list[int] = None,
        fused_dim: int = 256,
        dropout: float = 0.1,
        emb_size: int = 50,
    ):
        super().__init__()

        if branch_channels is None:
            branch_channels = [32, 64, 128]

        self.n_classes = n_classes

        # --- Branch 1: Raw signal ---
        self.raw_branch = RawSignalBranch(
            signal_length=signal_length,
            channels=branch_channels,
            dropout=dropout,
        )

        # --- Branch 2: ICA features ---
        self.has_ica = U is not None and M is not None
        if self.has_ica:
            self.ica_branch = ICABranch(
                U=U, M=M, m=m, s=s,
                channels=branch_channels,
                dropout=dropout,
            )

        # --- Branch 3: Fryze decomposition ---
        self.fryze_branch = FryzeBranch(
            signal_length=signal_length,
            emb_size=emb_size,
            channels=branch_channels,
            dropout=dropout,
        )

        # --- Branch 4: FFT features ---
        self.fft_branch = FFTBranch(
            signal_length=signal_length,
            channels=branch_channels,
            dropout=dropout,
        )

        # --- Fusion ---
        branch_dims = [
            self.raw_branch.out_dim,
            self.fryze_branch.out_dim,
            self.fft_branch.out_dim,
        ]
        if self.has_ica:
            branch_dims.insert(1, self.ica_branch.out_dim)

        self.fusion = AttentionFusion(
            branch_dims=branch_dims,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, n_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input signal tensor (batch, signal_length)
        Returns:
            logits: (batch, n_classes) — raw logits for BCEWithLogitsLoss
        """
        feats = []

        # Branch 1: Raw signal
        feats.append(self.raw_branch(x))

        # Branch 2: ICA
        if self.has_ica:
            feats.append(self.ica_branch(x))

        # Branch 3: Fryze
        feats.append(self.fryze_branch(x))

        # Branch 4: FFT
        feats.append(self.fft_branch(x))

        # Fuse all branches
        fused = self.fusion(feats)

        # Classify
        return self.classifier(fused)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# Lightweight variant for constrained hardware (4GB VRAM)
# ==============================================================================

class FusionResNetLite(FusionResNet):
    """
    Lighter version of Fusion-ResNet for machines with limited GPU memory.
    Uses fewer channels and a smaller fused dimension.
    Suitable for RTX 2050 (4GB) or similar.
    """

    def __init__(
        self,
        n_classes: int,
        signal_length: int = 400,
        U: np.ndarray = None,
        M: np.ndarray = None,
        m: np.ndarray = None,
        s: np.ndarray = None,
        dropout: float = 0.15,
        emb_size: int = 50,
    ):
        super().__init__(
            n_classes=n_classes,
            signal_length=signal_length,
            U=U, M=M, m=m, s=s,
            branch_channels=[16, 32, 64],
            fused_dim=128,
            dropout=dropout,
            emb_size=emb_size,
        )


# ==============================================================================
# Utility: Print model summary
# ==============================================================================

def model_summary(model: nn.Module, input_shape: tuple = (1, 400)):
    """Print a compact summary of the model."""
    print(f"{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*60}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total parameters:     {total_params:>12,}")
    print(f"Trainable parameters: {trainable_params:>12,}")
    print(f"Frozen parameters:    {frozen_params:>12,}")
    print(f"{'='*60}")

    # Estimate memory
    param_size_mb = total_params * 8 / (1024 ** 2)  # float64
    print(f"Param memory (fp64):  {param_size_mb:>10.1f} MB")
    print(f"Param memory (fp32):  {param_size_mb / 2:>10.1f} MB")

    # Forward pass memory estimate (rough)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    x = torch.randn(*input_shape, dtype=model_dtype, device=device)
    with torch.no_grad():
        try:
            _ = model(x)
            print(f"Forward pass:         OK")
        except Exception as e:
            print(f"Forward pass:         FAILED ({e})")
    print(f"{'='*60}")


if __name__ == '__main__':
    # Quick test with random data (no ICA)
    model = FusionResNet(n_classes=15, signal_length=400).double()
    model_summary(model, input_shape=(4, 400))

    print(f"\nTrainable parameters: {model.count_parameters():,}")

    # Test lite variant
    model_lite = FusionResNetLite(n_classes=15, signal_length=400).double()
    model_summary(model_lite, input_shape=(4, 400))

    print(f"\nLite trainable parameters: {model_lite.count_parameters():,}")

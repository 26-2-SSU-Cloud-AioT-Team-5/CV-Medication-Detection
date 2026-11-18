"""ResNet-18 pill-count classifier; output index is the count (0 through 7)."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(pretrained: bool = True) -> nn.Module:
    """Build an unfrozen ResNet-18 returning eight raw logits per RGB image.

    Use pretrained=False before loading a saved state_dict to avoid downloads.
    Device placement, preprocessing and training belong to the caller.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 8)
    return model

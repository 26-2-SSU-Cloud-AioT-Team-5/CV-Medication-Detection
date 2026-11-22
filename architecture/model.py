"""ResNet-18 models for pill count and frame status."""

from collections.abc import Mapping

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

COUNT_CLASSES = 8
STATUS_NAMES = ("countable", "hand_occlusion", "lid_closed")

#------------------------------

def build_model(pretrained: bool = True) -> nn.Module:
    """
    Build an unfrozen ResNet-18 returning eight raw logits per RGB image.

    Use pretrained=False before loading a saved state_dict to avoid downloads.
    Device placement, preprocessing and training belong to the caller.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, COUNT_CLASSES)

    return model

#------------------------------

class PillStateClassifier(nn.Module):
    """Return (count logits, status logits) from one pass through the backbone."""

    def __init__(self, count_state=None):
        super().__init__()

        backbone = build_model(pretrained=False)
        if count_state is not None:
            backbone.load_state_dict(count_state, strict=True)

        self.count_head = backbone.fc
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        self.status_head = nn.Linear(self.count_head.in_features, len(STATUS_NAMES))

    def forward(self, images):
        features = self.backbone(images)

        return self.count_head(features), self.status_head(features)
    
#------------------------------

def load_model_state(path):
    """
    Read either a raw state dict or a checkpoint containing model_state_dict.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected a state_dict or checkpoint dictionary: {path}")
    
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Empty or invalid model_state_dict: {path}")
    
    return state

"""ResNet-18 classifier for pill presence: not_filled=0, filled=1."""

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class PillClassifier(nn.Module):
    """Return one raw logit per image, with shape ``[batch_size]``.

    Train with BCEWithLogitsLoss. At inference, sigmoid(logits) gives the
    probability of the positive class (filled / pill present).
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, images: Tensor) -> Tensor:
        return self.backbone(images).squeeze(-1)
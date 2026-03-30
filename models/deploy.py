import torch.nn as nn
from yolox_face.losses.yolox_loss import YOLOXLoss


class DeploymentWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_helper = YOLOXLoss(num_classes=1)

    def forward(self, x):
        outputs = self.model(x)
        decoded = self.loss_helper.decode_outputs(outputs)
        return decoded["boxes"], decoded["obj"], decoded["cls"], decoded["lmk"]
# libs/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        if pred.shape[1] == 3:
            pred = 0.299*pred[:,0:1] + 0.587*pred[:,1:2] + 0.114*pred[:,2:3]
            target = 0.299*target[:,0:1] + 0.587*target[:,1:2] + 0.114*target[:,2:3]

        pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_y = F.conv2d(pred, self.sobel_y, padding=1)
        tgt_x = F.conv2d(target, self.sobel_x, padding=1)
        tgt_y = F.conv2d(target, self.sobel_y, padding=1)

        pred_mag = torch.sqrt(pred_x**2 + pred_y**2 + 1e-8)
        tgt_mag = torch.sqrt(tgt_x**2 + tgt_y**2 + 1e-8)

        return F.l1_loss(pred_mag, tgt_mag)

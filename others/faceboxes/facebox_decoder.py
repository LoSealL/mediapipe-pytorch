import torch
import torch.nn as nn


class FaceBoxesDecode(nn.Module):
    def __init__(self, n_classes, n_coords, anchors, scale):
        super().__init__()
        assert anchors.ndim == 2
        assert anchors.shape[1] == 4

        self.n_classes = n_classes
        self.n_boxes = anchors.shape[0]
        self.n_coords = n_coords
        self.scale = torch.tensor(scale, requires_grad=False)        
        self.anchors = anchors

    def decode_boxes(self, raw_boxes):
        scale = self.scale.to(raw_boxes.device)
        anchors = self.anchors.to(raw_boxes.device)[None, ...]
        xy = raw_boxes[..., :2] * anchors[..., 2:] * scale[0] + anchors[..., :2]
        wh = anchors[..., 2:] * torch.exp(raw_boxes[..., 2:] * scale[1])
        # cxywh
        r = torch.cat((xy, wh), dim=-1)
        return r

    def forward(self, scores, boxes):
        assert scores.ndim == 3
        assert boxes.ndim == 3
        assert scores.shape[1] == self.n_boxes
        assert scores.shape[2] == self.n_classes
        assert boxes.shape[1] == self.n_boxes
        assert boxes.shape[2] == self.n_coords

        boxes = self.decode_boxes(boxes)
        return scores[..., -1:], boxes

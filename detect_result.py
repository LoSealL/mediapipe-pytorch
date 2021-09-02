import torch
import torchvision


class DetectResult(torch.nn.Module):
    def __init__(self, n_classes, n_coords, min_score_threshold, nms_threshold):
        super(DetectResult, self).__init__()
        self.n_classes = n_classes
        self.n_coords = n_coords
        self.n_keypoints = (n_coords - 4) // 2
        self.min_score = min_score_threshold
        self.nms_threshold = nms_threshold

    def forward(self, score, box, cls=None):
        mask = score >= self.min_score
        mask.squeeze_(-1)
        box = box[mask]
        score = score[mask]
        if cls is None and self.n_classes == 1:
            label = torch.ones_like(score).long()
        else:
            assert cls.ndim == 2
            assert cls.shape[1] == self.n_classes
            cls = cls.max(-1)
            label = cls.indices[mask].long()
        bbox = box[..., :4]
        bbox = torchvision.ops.box_convert(bbox, 'cxcywh', 'xyxy')
        idx = torchvision.ops.batched_nms(bbox.float(), score.squeeze(-1), label.squeeze(-1), self.nms_threshold)
        return score[idx], box[idx]

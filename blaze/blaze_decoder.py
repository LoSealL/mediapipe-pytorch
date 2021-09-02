import numpy as np
import torch
import torch.nn as nn


def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) * 0.5
    return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)


def generate_anchors(options: dict):
    strides_size = len(options["strides"])
    assert options["num_layers"] == strides_size

    anchors = []
    layer_id = 0
    while layer_id < strides_size:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []

        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < strides_size) and \
                (options["strides"][last_same_stride_layer] == options["strides"][layer_id]):
            scale = calculate_scale(options["min_scale"],
                                    options["max_scale"],
                                    last_same_stride_layer,
                                    strides_size)

            if last_same_stride_layer == 0 and options["reduce_boxes_in_lowest_layer"]:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios.append(1.0)
                aspect_ratios.append(2.0)
                aspect_ratios.append(0.5)
                scales.append(0.1)
                scales.append(scale)
                scales.append(scale)
            else:
                for aspect_ratio in options["aspect_ratios"]:
                    aspect_ratios.append(aspect_ratio)
                    scales.append(scale)

                if options["interpolated_scale_aspect_ratio"] > 0.0:
                    scale_next = 1.0 if last_same_stride_layer == strides_size - 1 \
                        else calculate_scale(options["min_scale"],
                                             options["max_scale"],
                                             last_same_stride_layer + 1,
                                             strides_size)
                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios.append(options["interpolated_scale_aspect_ratio"])

            last_same_stride_layer += 1

        for i in range(len(aspect_ratios)):
            ratio_sqrts = np.sqrt(aspect_ratios[i])
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options["strides"][layer_id]
        feature_map_height = int(np.ceil(options["input_size_height"] / stride))
        feature_map_width = int(np.ceil(options["input_size_width"] / stride))

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options["anchor_offset_x"]) / feature_map_width
                    y_center = (y + options["anchor_offset_y"]) / feature_map_height

                    new_anchor = [x_center, y_center, 0, 0]
                    if options["fixed_anchor_size"]:
                        new_anchor[2] = 1.0
                        new_anchor[3] = 1.0
                    else:
                        new_anchor[2] = anchor_width[anchor_id]
                        new_anchor[3] = anchor_height[anchor_id]
                    anchors.append(new_anchor)

        layer_id = last_same_stride_layer

    return anchors


class BlazeDecodeBox(nn.Module):
    SCORE_NEED_SIGMOID = True

    def __init__(self, n_classes, n_boxes, n_coords, x_scale, y_scale, w_scale, h_scale, anchors):
        super(BlazeDecodeBox, self).__init__()
        self.n_classes = n_classes
        self.n_boxes = n_boxes
        self.n_coords = n_coords
        scale = [x_scale, y_scale, w_scale, h_scale]
        while len(scale) < n_coords:
            scale.extend((x_scale, y_scale))
        self.scale = torch.tensor(scale)
        self.scale.requires_grad = False
        assert anchors.ndim == 2
        assert anchors.shape[0] == self.n_boxes
        assert anchors.shape[1] == 4
        self.anchor_scale = anchors[..., 2:4].repeat([1, self.n_coords // 2])[None]
        self.anchor_offset = anchors[..., 0:2].repeat([1, self.n_coords // 2])
        self.anchor_offset[..., 2:4] = 0
        self.anchor_offset = self.anchor_offset[None]

    def decode_boxes(self, raw_boxes):
        scale = self.scale.to(raw_boxes.device)
        anchor_scale = self.anchor_scale.to(raw_boxes.device)
        anchor_offset = self.anchor_offset.to(raw_boxes.device)
        r = raw_boxes / scale
        r *= anchor_scale
        r += anchor_offset
        return r

    def forward(self, scores, boxes):
        if self.SCORE_NEED_SIGMOID:
            scores.sigmoid_()

        assert scores.ndim == 3
        assert boxes.ndim == 3
        assert scores.shape[1] == self.n_boxes
        assert scores.shape[2] == self.n_classes
        assert boxes.shape[1] == self.n_boxes
        assert boxes.shape[2] == self.n_coords

        boxes = self.decode_boxes(boxes)
        return scores, boxes

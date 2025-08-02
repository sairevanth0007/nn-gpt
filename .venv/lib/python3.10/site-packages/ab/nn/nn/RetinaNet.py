import math
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
from collections import OrderedDict
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision.ops import boxes as box_ops
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.ops import sigmoid_focal_loss
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection import _utils as det_utils


def supported_hyperparameters():
    return {'lr', 'momentum', 'fg_iou_thresh', 'bg_iou_thresh', 'score_thresh',
            'nms_thresh', 'detections_per_img', 'topk_candidates', 'pretrained'}


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


def _default_anchorgen():
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return anchor_generator


class RetinaNetHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, norm_layer=None):
        super().__init__()
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes, norm_layer=norm_layer)
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors, norm_layer=norm_layer)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        return {
            "classification": self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            "bbox_regression": self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs)
        }

    def forward(self, x):
        return {
            "cls_logits": self.classification_head(x),
            "bbox_regression": self.regression_head(x)
        }


class RetinaNetClassificationHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01, norm_layer=None):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(Conv2dNormActivation(in_channels, in_channels, norm_layer=norm_layer))
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.BETWEEN_THRESHOLDS = -1

    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []
        cls_logits = head_outputs["cls_logits"]

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]]
            ] = 1.0

            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            losses.append(
                sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    reduction="sum",
                ) / max(1, num_foreground)
            )

        return _sum(losses) / len(targets)

    def forward(self, x):
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    def __init__(self, in_channels, num_anchors, norm_layer=None):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(Conv2dNormActivation(in_channels, in_channels, norm_layer=norm_layer))
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

        self._loss_type = "l1"

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        losses = []
        bbox_regression = head_outputs["bbox_regression"]

        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
                targets, bbox_regression, anchors, matched_idxs
        ):
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            matched_gt_boxes_per_image = targets_per_image["boxes"][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            target_regression = self.box_coder_encode_single(matched_gt_boxes_per_image, anchors_per_image)
            losses.append(torch.nn.functional.l1_loss(bbox_regression_per_image, target_regression, reduction="sum") / max(1, num_foreground))

        return _sum(losses) / len(targets)

    def forward(self, x):
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)

    def box_coder_encode_single(self, reference_boxes, proposals):
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor([1.0, 1.0, 1.0, 1.0], dtype=dtype, device=device)

        wx = weights[0]
        wy = weights[1]
        ww = weights[2]
        wh = weights[3]

        proposals_x1 = proposals[:, 0].unsqueeze(1)
        proposals_y1 = proposals[:, 1].unsqueeze(1)
        proposals_x2 = proposals[:, 2].unsqueeze(1)
        proposals_y2 = proposals[:, 3].unsqueeze(1)

        reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
        reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
        reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
        reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

        ex_widths = proposals_x2 - proposals_x1
        ex_heights = proposals_y2 - proposals_y1
        ex_ctr_x = proposals_x1 + 0.5 * ex_widths
        ex_ctr_y = proposals_y1 + 0.5 * ex_heights

        gt_widths = reference_boxes_x2 - reference_boxes_x1
        gt_heights = reference_boxes_y2 - reference_boxes_y1
        gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
        gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def box_coder_decode_single(self, rel_codes, boxes):
        boxes = boxes.to(dtype=rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = [1.0] * 4
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        dw = torch.clamp(dw, max=math.log(1000.0 / 16))
        dh = torch.clamp(dh, max=math.log(1000.0 / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        num_classes = out_shape[0]
        fg_iou_thresh = prm['fg_iou_thresh']
        bg_iou_thresh = prm['bg_iou_thresh']

        self.score_thresh = prm['score_thresh']
        self.nms_thresh = prm['nms_thresh']
        self.detections_per_img = int(600 * prm['detections_per_img']) + 1
        self.topk_candidates = int(2000 * prm['topk_candidates']) + 1

        use_pretrained = prm['pretrained'] > 0.5
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
        backbone = _resnet_fpn_extractor(
            backbone,
            trainable_layers=3 if use_pretrained else 5,
            returned_layers=[2, 3, 4],
            extra_blocks=LastLevelP6P7(256, 256)
        )


        anchor_generator = _default_anchorgen()
        norm_layer = partial(nn.GroupNorm, 32)

        self.backbone = backbone
        self.anchor_generator = anchor_generator
        self.head = RetinaNetHead(
            backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0],
            num_classes,
            norm_layer=norm_layer
        )

        self.transform = GeneralizedRCNNTransform(
            min_size=320,
            max_size=320,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
        self.num_classes = num_classes

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

    def forward(self, images, targets=None):
        if targets is not None:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise ValueError("Expected target boxes to be of type Tensor.")

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        images = self.transform(images)[0]
        features = self.backbone(images.tensors)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())

        head_outputs = self.head(features)
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections = []

        if self.training:
            losses = self.compute_loss(targets, head_outputs, anchors)
        else:
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                self._has_warned = True
            return losses, detections

        return losses if self.training else detections

    def compute_loss(self, targets, head_outputs, anchors):
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(self, head_outputs, anchors, image_sizes):
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]

        num_images = len(image_sizes)
        detections = []

        for index in range(num_images):
            box_regression_per_image = box_regression[index]
            logits_per_image = class_logits[index]
            anchors_per_image = anchors[index]
            image_size = image_sizes[index]

            scores_per_image = torch.sigmoid(logits_per_image)

            # Keep only the top k scores
            top_k = min(self.topk_candidates, scores_per_image.shape[0])
            scores_per_image, topk_idxs = scores_per_image.flatten().topk(top_k)

            anchor_idxs = topk_idxs // self.num_classes
            labels_per_image = topk_idxs % self.num_classes

            boxes_per_image = self.head.regression_head.box_coder_decode_single(
                box_regression_per_image[anchor_idxs], anchors_per_image[anchor_idxs]
            )
            boxes_per_image = box_ops.clip_boxes_to_image(boxes_per_image, image_size)

            keep = box_ops.batched_nms(
                boxes_per_image,
                scores_per_image,
                labels_per_image,
                self.nms_thresh,
            )
            keep = keep[:self.detections_per_img]

            detections.append({
                "boxes": boxes_per_image[keep],
                "labels": labels_per_image[keep],
                "scores": scores_per_image[keep],
            })

        return detections

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)

            # labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]
            labels = labels.to(self.device)
            self.optimizer.zero_grad()  # Changed from optimizer to self.optimizer

            losses = self(inputs, labels)  # Changed from forward_pass to self()
            loss = sum(loss for loss in losses.values())

            loss.backward()
            self.optimizer.step()  # Changed from optimizer to self.optimizer

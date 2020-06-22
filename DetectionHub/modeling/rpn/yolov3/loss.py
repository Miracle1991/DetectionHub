# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from .utils import concat_box_prediction_layers_yolov3

from ...balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from DetectionHub.layers import smooth_l1_loss
from DetectionHub.modeling.matcher import Matcher
from DetectionHub.structures.boxlist_ops import boxlist_iou
from DetectionHub.structures.boxlist_ops import cat_boxlist


class YOLOV3LossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, box_coder,
                 generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        cls_labels = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            cls_label_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0
            cls_label_per_image[bg_indices] = 0

            # # discard anchors that go out of the boundaries of the image
            # if "not_visibility" in self.discard_cases:
            #     labels_per_image[~anchors_per_image.get_field("visibility")] = -1
            #
            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1
                cls_label_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            cls_labels.append(cls_label_per_image)

        return labels, regression_targets, cls_labels


    def __call__(self, anchors, objectness, cls, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets, cls_labels = self.prepare_targets(anchors, targets)

        objectness, cls, box_regression = concat_box_prediction_layers_yolov3(objectness, box_regression, cls)

        objectness = objectness.squeeze()

        obj_labels = torch.cat(labels, dim=0)
        cls_labels = torch.cat(cls_labels, dim=0).long()

        sampled_pos_inds = torch.nonzero(obj_labels == 1).squeeze(1)
        sampled_neg_inds = torch.nonzero(obj_labels == 0).squeeze(1)

        # sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (max(1, sampled_pos_inds.numel()))

        cls = cls[sampled_pos_inds]

        # a = {}
        # for c in cls_labels:
        #     if not str(c.cpu().numpy()) in a.keys():
        #         a[str(c.cpu().numpy())] = 0
        #     a[str(c.cpu().numpy())] += 1

        cls_labels = cls_labels[sampled_pos_inds] - 1

        cls_loss = F.cross_entropy(cls, cls_labels)

        objectness_pos_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_pos_inds], obj_labels[sampled_pos_inds]
        )


        objectness_neg_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_neg_inds], obj_labels[sampled_neg_inds]
        )

        objectness_loss = objectness_pos_loss + objectness_neg_loss * 100
        # predict = objectness[sampled_inds].view(-1, 1)
        # target = labels[sampled_inds].view(-1, 1)
        #
        # predict = torch.sigmoid(predict)
        # logpt = torch.log(predict)
        #
        # lognpt = torch.log(1 - predict)
        #
        # loss_positive = -1 * 0.5 * target * logpt * (1 - predict) ** 2
        # loss_negtive = -1 * (1 - 0.5) * (1 - target) * lognpt * predict ** 2
        #
        # focal_loss = (loss_positive + loss_negtive).mean()

        return objectness_loss, box_loss, cls_loss

# This function should be overwritten in RetinaNet
def generate_yolo_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_yolov3_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.YOLOV3.FG_IOU_THRESHOLD,
        cfg.MODEL.YOLOV3.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    loss_evaluator = YOLOV3LossComputation(
        matcher,
        box_coder,
        generate_yolo_labels
    )
    return loss_evaluator


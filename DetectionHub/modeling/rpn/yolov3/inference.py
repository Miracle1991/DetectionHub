# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from DetectionHub.modeling.box_coder import BoxCoder
from DetectionHub.structures.bounding_box import BoxList
from DetectionHub.structures.boxlist_ops import cat_boxlist
from DetectionHub.structures.boxlist_ops import boxlist_nms
from DetectionHub.structures.boxlist_ops import remove_small_boxes

from ..utils import cat
from .utils import permute_and_flatten

class YOLOv3PostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
        fpn_post_nms_per_batch=True,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(YOLOv3PostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.fpn_post_nms_per_batch = fpn_post_nms_per_batch

        self.onnx_export = False

    def prepare_onnx_export(self):
        self.onnx_export = True

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression, cls):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape

        ###
        # show heat map
        ###
        # import matplotlib.pyplot as plt
        # import cv2
        # import numpy as np
        # img = cv2.imread("/home/w/workspace/onnx/maskrcnn-benchmark/demo/test_yolo.jpg")
        # img = cv2.resize(img, (416, 416))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # temp = objectness[:, 0].cpu()[0].numpy() * 255
        # temp = temp.astype(np.uint8)
        # temp = cv2.resize(temp, (416, 416))
        # img = cv2.addWeighted(img, 0.5, temp, 0.5, 1)
        #
        # plt.imshow(img)
        # plt.show()

        ###
        # show heat map end
        ###


        N, AXC, H, W = cls.shape

        C = int(AXC/A)

        # put in the same format as anchors
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        objectness = objectness.sigmoid()

        cls = permute_and_flatten(cls, N, A, C, H, W)

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)

        num_anchors = A * H * W

        if self.onnx_export:
            from torch.onnx import operators
            num_anchors = operators.shape_as_tensor(objectness)[1].unsqueeze(0)

            pre_nms_top_n = torch.min(
                torch.cat(
                    (torch.tensor([self.pre_nms_top_n], dtype=torch.long),
                     num_anchors), 0))
        else:
            pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

        batch_idx = torch.arange(N, device=device)[:, None]
        if self.onnx_export:
            # NOTE: for now only batch == 1 is supported for ONNX export.
            assert topk_idx.size(0) == 1
            topk_idx = topk_idx.squeeze(0)
            box_regression = box_regression.index_select(1, topk_idx)
        else:
            box_regression = box_regression[batch_idx, topk_idx]
            cls = cls[batch_idx, topk_idx]

        image_shapes = [box.size for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        if self.onnx_export:
            concat_anchors = concat_anchors.reshape(N, -1, 4).index_select(1, topk_idx)
        else:
            concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4)
        )

        proposals = proposals.view(N, -1, 4)
        cls = torch.argmax(cls, -1) + 1
        result = []
        for proposal, score, c, im_shape in zip(proposals, objectness, cls, image_shapes):
            boxlist = BoxList(proposal, im_shape, mode="xyxy")
            boxlist.add_field("scores", score)
            boxlist.add_field("labels", c)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size, self.onnx_export)
            boxlist = boxlist_nms(
                boxlist,
                self.nms_thresh,
                max_proposals=self.post_nms_top_n,
                score_field="scores",
            )
            result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, cls, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b, c in zip(anchors, objectness, box_regression, cls):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b, c))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # NOTE: it should be per image, and not per batch. However, to be consistent
        # with Detectron, the default is per batch (see Issue #672)
        if self.training and self.fpn_post_nms_per_batch:
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("scores")

                if self.onnx_export:
                    from torch.onnx import operators
                    objectness_len = operators.shape_as_tensor(objectness)
                    post_nms_top_n = torch.min(
                        torch.cat(
                            (torch.tensor([self.fpn_post_nms_top_n], dtype=torch.long),
                             objectness_len), 0))
                else:
                    post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))

                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_yolov3_post_processor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST

    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    fpn_post_nms_per_batch = config.MODEL.RPN.FPN_POST_NMS_PER_BATCH
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = YOLOv3PostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        fpn_post_nms_per_batch=fpn_post_nms_per_batch,
    )
    return box_selector

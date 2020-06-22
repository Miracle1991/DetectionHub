# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from DetectionHub.modeling.rpn.yolov3.yolov3_feature_extractors import build_yolov3_feature_extractors
from DetectionHub.modeling.rpn.yolov3.yolov3_predictor import make_yolov3_predictor
from DetectionHub.modeling.rpn.yolov3.loss import make_yolov3_loss_evaluator
from ..anchor_generator import make_anchor_generator_yolov3
from DetectionHub.modeling.box_coder import BoxCoder
from .inference import make_yolov3_post_processor
import cv2
import numpy  as np

class Yolov3Head(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(Yolov3Head, self).__init__()
        self.feature_extractor = build_yolov3_feature_extractors(cfg)
        self.predictor1 = make_yolov3_predictor(cfg)
        self.predictor2 = make_yolov3_predictor(cfg)
        self.predictor3 = make_yolov3_predictor(cfg)
        anchor_generator = make_anchor_generator_yolov3(cfg)
        self.anchor_generator = anchor_generator
        yolov3_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.loss_evaluator = make_yolov3_loss_evaluator(cfg, yolov3_box_coder)
        box_selector_test = make_yolov3_post_processor(cfg, yolov3_box_coder, is_train=False)
        self.box_selector_test = box_selector_test

    def forward(self, images, features, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # extract features that will be fed to the final classifier.
        x = self.feature_extractor(features)

        anchors = self.anchor_generator(images, x)

        # for xx, a in zip(x, anchors[0]):
        # bbox = anchors[0][2].bbox.cpu().numpy()
        # for b in bbox:
        #     img = cv2.imread('/home/w/workspace/onnx/maskrcnn-benchmark/demo/test_yolo.jpg')
        #     img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
        #     x1, y1 = int(max(0, b[0])), int(max(0, b[1]))
        #     x2, y2 = int(min(415, b[2])), int(min(415, b[3]))
        #     cv2.rectangle(img, (x1, y1), (x2, y2), 255, 2)
        #     cv2.imshow("", img)
        #     cv2.waitKey(0)

        objectness1, rpn_box_regression1, cls1 = self.predictor1(x[0])
        objectness2, rpn_box_regression2, cls2 = self.predictor2(x[1])
        objectness3, rpn_box_regression3, cls3 = self.predictor3(x[2])


        if self.training:
            return self._forward_train(anchors,
                                       [objectness1, objectness2, objectness3],
                                       [rpn_box_regression1, rpn_box_regression2, rpn_box_regression3],
                                       [cls1, cls2, cls3],
                                       targets)
        else:
            return self._forward_test(anchors,
                                      [objectness1, objectness2, objectness3],
                                      [rpn_box_regression1, rpn_box_regression2, rpn_box_regression3],
                                      [cls1, cls2, cls3])

    def _forward_train(self, anchors, objectness, rpn_box_regression, cls, targets):
        objectness_loss, box_loss, cls_loss = self.loss_evaluator(
            anchors,
            objectness,
            rpn_box_regression,
            cls,
            targets
        )

        return (
            None,
            {},
            dict(loss_objectness=objectness_loss, loss_classifier=cls_loss, loss_box_reg=box_loss,)
        )

    def _forward_test(self, anchors, objectness, rpn_box_regression, cls):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression, cls)
        # high-to-low confidence order.
        inds = [
            box.get_field("scores").sort(descending=True)[1] for box in boxes
        ]
        boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return None, boxes, {}


def build_yolov3_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return Yolov3Head(cfg, in_channels)

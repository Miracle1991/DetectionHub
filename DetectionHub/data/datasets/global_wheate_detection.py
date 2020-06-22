# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
from DetectionHub.structures.bounding_box import BoxList
from DetectionHub.structures.segmentation_mask import SegmentationMask
from DetectionHub.structures.keypoint import PersonKeypoints
import cv2
from PIL import Image
import os
import pandas as pd
import re
import numpy as np

class WheatDataset(object):
    def __init__(self, data_dir, transforms=None):
        super().__init__()
        DIR_INPUT = '/media/w/Data/globel-wheat-detection'
        self.image_dir = f'{DIR_INPUT}/train'
        train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
        train_df['x'] = -1
        train_df['y'] = -1
        train_df['w'] = -1
        train_df['h'] = -1

        def expand_bbox(x):
            r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
            if len(r) == 0:
                r = [-1, -1, -1, -1]
            return r

        train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
        train_df.drop(columns=['bbox'], inplace=True)
        train_df['x'] = train_df['x'].astype(np.float)
        train_df['y'] = train_df['y'].astype(np.float)
        train_df['w'] = train_df['w'].astype(np.float)
        train_df['h'] = train_df['h'].astype(np.float)

        image_ids = train_df['image_id'].unique()
        train_ids = image_ids
        self.train_df = train_df[train_df['image_id'].isin(train_ids)]

        images_root = '/media/w/Data/globel-wheat-detection/train'
        self.images = os.listdir(images_root)
        self._transforms = transforms

    def __getitem__(self, index):
        while(True):
            image_id = self.images[index].split('.')[0]
            records = self.train_df[self.train_df['image_id'] == image_id]

            img = Image.open(f'{self.image_dir}/{image_id}.jpg').convert("RGB")

            boxes = records[['x', 'y', 'w', 'h']].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes

            if len(boxes) >= 1:
                break
            else:
                index = index - 1

        target = BoxList(boxes, img.size, mode="xyxy")

        classes = [1 for box in boxes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, index

    def __len__(self) -> int:
        return len(self.images)

    def get_img_info(self, index):
        return {"height": 1024, "width": 1024}
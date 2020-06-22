import csv
import cv2
import os
import pandas as pd
import re
import numpy as np

label = {}

DIR_INPUT = '/media/w/Data/globel-wheat-detection'
DIR_TRAIN = f'{DIR_INPUT}/train'
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
train_df = train_df[train_df['image_id'].isin(train_ids)]

images_root = '/media/w/Data/globel-wheat-detection/train'
images = os.listdir(images_root)

for image in images:
    image_id = image.split('.')[0]
    img = cv2.imread(os.path.join(images_root, image))
    labels = train_df[train_df['image_id'] == image_id]
    boxes = labels[['x', 'y', 'w', 'h']].values
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    target = {}
    target['boxes'] = boxes
    target['labels'] = labels
    for bbox in boxes:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
    cv2.imshow('', img)
    cv2.waitKey(0)

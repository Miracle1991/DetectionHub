# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import csv
import argparse
import cv2

from DetectionHub.config import cfg
from demo.predictor import COCODemo
import os
import time

def resize2_1024(point):
    point = point/600*1024
    point = [int(p) for p in point]
    return point

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="/home/w/workspace/DetectionHub/configs/global_wheat.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=0.5,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    result_dict = {}
    image_dir = '/media/w/Data/globel-wheat-detection/test'
    image_list = os.listdir(image_dir)
    for i in image_list:
        image = cv2.imread(os.path.join(image_dir, i))
        start_time = time.time()
        image = cv2.resize(image, (600, 600), cv2.INTER_CUBIC)
        composite, predictions = coco_demo.run_on_opencv_image(image)
        result = []
        for bbox, score in zip(predictions.bbox, predictions.get_field("scores")):
            bbox = bbox.cpu().numpy()
            score = score.cpu().numpy()
            bbox = resize2_1024(bbox)
            result.append(str(score))
            for b in bbox:
                result.append(str(b))

        result_dict[i.split('.')[0]] = ' '.join(result)

        print("Time: {:.2f} s / img".format(time.time() - start_time))
        # cv2.imshow("COCO detections", composite)
        # if cv2.waitKey(0) == 27:
        #     break  # esc to quit
    with open("test.csv", "w") as csvfile:
        writer = csv.writer(csvfile)

        # 先写入columns_name
        writer.writerow(["image_id", "PredictionString"])
        for key in result_dict.keys():
            writer.writerow([key, result_dict[key]])
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
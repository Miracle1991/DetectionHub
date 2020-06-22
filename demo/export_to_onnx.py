import os
import numpy

from io import BytesIO

from matplotlib import pyplot

import requests
import torch
import timeit

from PIL import Image
from DetectionHub.config import cfg
from predictor import COCODemo
from DetectionHub.structures.image_list import ImageList

import pytorch_export_patch

if __name__ == "__main__":
    # load config from file and command-line arguments
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg.merge_from_file(
        os.path.join(project_dir,
                     "configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml"))
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=480,
    )

    # prepare for onnx export
    coco_demo.model.prepare_onnx_export()


def single_image_to_top_predictions(image):
    image_list = ImageList(image.unsqueeze(0), [(int(image.size(-2)), int(image.size(-1)))])

    for param in coco_demo.model.parameters():
        param.requires_grad = False

    result, = coco_demo.model(image_list)
    scores = result.get_field("scores")
    result = (result.bbox, result.get_field('labels'), scores)
    return result


class FRCNNModel(torch.nn.Module):
    def forward(self, image):
        return single_image_to_top_predictions(image)


def fetch_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


if __name__ == '__main__':
    pil_image = fetch_image(
        url="http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")

    """
    Preprocessing image.
    """
    # convert to BGR format
    image = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
    original_image = image
    image = torch.nn.functional.upsample(image.permute(2, 0, 1).unsqueeze(0).to(torch.float), size=(960, 1280)).to(torch.uint8).squeeze(0).permute(1, 2, 0).to('cpu')

    image = image.permute(2, 0, 1)

    if not cfg.INPUT.TO_BGR255:
        image = image.float() / 255.0
        image = image[[2, 1, 0]]
    else:
        image = image.float()

    # we absolutely want fixed size (int) here (or we run into a tracing error (or bug?)
    # or we might later decide to make things work with variable size...
    image = image - torch.tensor(cfg.INPUT.PIXEL_MEAN)[:, None, None].to('cpu')

    model_path = 'faster_rcnn_R_50_FPN_1x.onnx'

    with torch.no_grad():
        model = FRCNNModel()
        model.eval()

        torch.onnx.export(model, (image,), model_path, verbose=True, opset_version=10, strip_doc_string=True, do_constant_folding=True)
        print("export done!")
        pytorch_export_patch.postprocess_model(model_path)

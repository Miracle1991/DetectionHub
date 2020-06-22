"""
Verified against

    pytorch-nightly: 1.2.0.dev20190604-py3.7_cuda10.0.130_cudnn7.5.1_0 pytorch
    torchvision:     from source 04188377c54aa9073e4c2496ddd9996da9fda629
    onnx:            1.5.0
    onnxruntime:     0.4.0
"""
import onnxruntime
import onnx
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def preprocess(image):
    # Resize
    ratio = 960.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image

print('##################################')
model = onnx.load('faster_rcnn_R_50_FPN_1x.onnx')
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)
print(model.graph)
print('##################################')
#
classes = [line.rstrip('\n') for line in open('coco_classes.txt')]

session = onnxruntime.InferenceSession('faster_rcnn_R_50_FPN_1x.onnx')
# session = onnxruntime.InferenceSession('/home/w/Downloads/myTensorRT/freespace_onnx/freespace.onnx')

img = Image.open('frcnn_demo.jpg')

"""
Preprocessing
"""
img_data = preprocess(img)



"""
Inference
"""
boxes, labels, scores = session.run(None, {
    session.get_inputs()[0].name: img_data
})

print(boxes.shape)
print(labels.shape)
print(scores.shape)

"""
Postprocessing
"""
def display_objdetect_image(image, boxes, labels, scores, score_threshold=0.7):
    # Resize boxes
    ratio = 960.0 / min(image.size[0], image.size[1])
    boxes /= ratio

    _, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(image)

    # Showing boxes with score > 0.7
    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
            ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='w', fontsize=12)
            ax.add_patch(rect)
    plt.show()

display_objdetect_image(img, boxes, labels, scores)
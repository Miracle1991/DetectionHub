Based on [existing work](https://github.com/facebookresearch/maskrcnn-benchmark/pull/138) of enabling Tracing/Scripting for MaskRCNN/FasterRCNN models, this branch extends the support for exporting FasterRCNN to ONNX.

With this patch, we are able to export [e2e_faster_rcnn_R-50-FPN_1x](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml) to ONNX, and run in ONNXRuntime backend.

## Setup Environment

Most parts are the same with [INSTALL.md](../INSTALL.md). We want to use pytorch-1.2 nightly with latest torchvision, so we build torchvision from source.

```bash
conda create --name maskrcnn_onnx
conda activate maskrcnn_onnx

conda install ipython

# DetectionHub and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python requests

# onnx
pip install onnx

# onnxruntime as onnx backend
pip install onnxruntime

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.0
conda install -c pytorch pytorch-nightly cudatoolkit=10.0

export INSTALL_DIR=$PWD

# install torchvision
cd $INSTALL_DIR
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py develop

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection - ONNX branch
cd $INSTALL_DIR
git clone https://github.com/bowenbao/maskrcnn-benchmark.git maskrcnn-onnx
cd maskrcnn-onnx
git checkout onnx_stage

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

unset INSTALL_DIR
```

## Export & Inference in ONNX
Under folder [demo](.), there are two python demos.

* demo/export_to_onnx.py:
  - Usage: python export_to_onnx.py
  - Exports [e2e_faster_rcnn_R-50-FPN_1x](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml) to faster_rcnn_R_50_FPN_1x.onnx.
* demo/eval_onnx.py:
  - Usage: python eval_onnx.py
  - Loads the exported ONNX model in previous section, and inference with ONNXRuntime.

## TODOs
* This branch depends on many updates to PyTorch ONNX exporter, many are still in PR under review. Once they are in PyTorch master we can remove the ```pytorch_export_patch.py``` patch.
* Add support for other configs and MaskRCNN models.
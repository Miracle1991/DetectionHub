import os

import torch

torch.ops.load_library(os.path.join(os.path.dirname(__file__), 'lib', 'libDetectionHub_customops.so'))

import os

# Force a safe matplotlib backend for non-notebook Python
os.environ['MPLBACKEND'] = 'Agg'

import sys
import torch

print("sys.path (first few):", sys.path[:5])
print("Torch version:", torch.__version__)
print("CUDA available?", torch.cuda.is_available())

# Now import mmdet3d (this may indirectly import matplotlib via mmdet)
import mmdet3d
from mmdet3d.apis import init_model

print("mmdet3d version:", mmdet3d.__version__)
print("Import OK!")

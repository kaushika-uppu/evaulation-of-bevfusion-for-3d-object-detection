# HPC A40 GPU Setup
This file describes the environment and module setup that was carried out to run BEVFusion on the HPC A40 GPU.

## Conda

    conda create -y -n py310 python=3.10
    conda activate py310

## Enable Internet (for downloads)
    export http_proxy=http://172.16.1.2:3128
    export https_proxy=http://172.16.1.2:3128

## Installations/Imports

    module load nvhpc-hpcx-cuda12/24.11 #load cuda module
    
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install mmengine
    
    python -m pip install -U pip setuptools wheel
    python -m pip install -U openmim

### mmcv

    Using pre-built wheels caused kernel errors when running inference --> building MMCV from scratch.
    export CUDA_HOME=/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6
    export CPATH=$CPATH:$CUDA_HOME/targets/x86_64-linux/include
    export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_HOME/targets/x86_64-linux/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/targets/x86_64-linux/lib
    export PATH=$CUDA_HOME/bin:$PATH
    
    export MATH_LIB_HOME=/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/math_libs/12.6/targets/x86_64-linux
    export CPATH=$CPATH:$MATH_LIB_HOME/include
    export LIBRARY_PATH=$LIBRARY_PATH:$MATH_LIB_HOME/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MATH_LIB_HOME/lib

Now, to build:

    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1
    FORCE_CUDA=1
    pip install -e . -v # compile with CUDA

### mmdetection

    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    pip install -e . -v --no-build-isolation
   
### mmdetection3d

    git clone https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    pip install -e . -v --no-build-isolation
 
### numpy
    pip uninstall numpy
    pip install 'numpy<2'

## Dataset
Downloaded NuScenes mini v1.0 from the NuScenes website:
https://www.nuscenes.org/download

Then, uploaded to HPC server and unpackaged:

    # upload to HPC
    scp ~/Downloads/v1.0-mini.tgz 014756859@coe-hpc1:~/3d_obj_det/mmdetection3d/data/
    
    # unpackage
    tar -xvz v1.0-mini.tgz -C data/nuscenes

## Model
To get model configuration and checkpoint, downloaded directly from BEVFusion Github repository:
https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion

Then, uploaded both to server:

    scp ./Downloads/bevfusion_mmdet3d.pth 014756859@coe-hpc1:~/3d_obj_det/mmdetection3d/checkpoints/
    scp ./Downloads/bevfusion_mini@coe-hpc1:~/3d_obj_det/mmdetection3d/projects/BEVFusion/configs


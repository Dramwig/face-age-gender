# 基于深度学习的性别与年龄的智能识别系统

基于yolo8-face和SSR-Net的轻量化性别与年龄识别框架

## Envs

```bash
conda create -n yolo python=3.9
conda activate yolo
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics # Install the ultralytics package from PyPI
pip install tensorflow==2.10.0 # For SSR-Net
```

## Use

yolo8-face: https://github.com/lindevs/yolov8-face?tab=readme-ov-file
SSR-Net: https://github.com/diovisgood/agender
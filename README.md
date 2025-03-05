# Deep Learning Based Gender and Age Recognition System

基于 YOLOv8-face 和 SSR-Net 的实时人脸性别与年龄识别系统。使用轻量级模型实现快速准确的人脸检测和属性分析，支持视频流处理。

A real-time face gender and age recognition system based on YOLOv8-face and SSR-Net. Using lightweight models to achieve fast and accurate face detection and attribute analysis, supporting video stream processing.

<p align="center">
  <img src="outputs/output.gif" alt="e.q."/>
</p>

## Architecture

- Face Detection: YOLOv8-face for robust face detection
- Gender Recognition: SSR-Net based gender classifier
- Age Estimation: SSR-Net based age regressor
- Video Processing: OpenCV based video stream handler

## Envs

```bash
conda create -n yolo python=3.9
conda activate yolo
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics # Install the ultralytics package from PyPI
pip install tensorflow==2.10.0 # For SSR-Net
```

## Basis 

- yolo8-face: https://github.com/lindevs/yolov8-face?tab=readme-ov-file
- SSR-Net: https://github.com/diovisgood/agender
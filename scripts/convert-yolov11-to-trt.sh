#!/bin/bash -e

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# install ultralyics package
pip install --upgrade ultralytics
# pip install ultralytics torch torchvision torchaudio

mkdir -p $SCRIPT_PATH/../build/models
cd $SCRIPT_PATH/../build/models

wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt

python3 $SCRIPT_PATH/export-yolo-onnx.py

Convert each ONNX model to TensorRT engine
for model in yolo11n yolo11s yolo11m yolo11l yolo11x; do
    echo "Converting ${model}.onnx to ${model}.engine..."
    $SCRIPT_PATH/../build/YOLOv11TRT convert ${model}.onnx ${model}.engine
    echo "Completed conversion of ${model}"
done
from ultralytics import YOLO

# List of YOLO11 models to export
models = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]

# Export each model to ONNX format
for model_name in models:
    print(f"Exporting {model_name}.pt to ONNX...")
    model = YOLO(f"{model_name}.pt")
    export_path = model.export(format="onnx")
    print(f"Exported {model_name} to: {export_path}")
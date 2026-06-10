"""Fine-tune YOLOv8-nano to detect DataMatrix codes on the synthetic dataset.
Spike: weights/runs go to /tmp (not the repo).

    .venv/bin/python -m tools.train_yolo [epochs]
"""
import sys

from ultralytics import YOLO


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    m = YOLO("yolov8n.pt")               # COCO-pretrained nano, fine-tune
    m.train(data="/tmp/dm_yolo/data.yaml", epochs=epochs, imgsz=640, batch=16,
            device="mps", project="/tmp/dm_yolo_runs", name="dm", patience=12,
            plots=False, verbose=True)


if __name__ == "__main__":
    main()

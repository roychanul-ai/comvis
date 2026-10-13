
from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "indoor" / "data.yaml"
OUTPUT_DIR = ROOT / "outputs" / "training"

def main():
    print("PyTorch:", torch.__version__)
    print("Device: CPU")
    print("Dataset:", DATA_YAML)

    # Membuat model dari konfigurasi, tanpa pretrained weights.
    model = YOLO("yolo26n.yaml")

    results = model.train(
        data=str(DATA_YAML),
        epochs=20,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        pretrained=False,
        project=str(OUTPUT_DIR),
        name="indoor_yolo26n_scratch",
        exist_ok=True,
        patience=5,
        save=True,
        plots=True,
        verbose=True,
    )

    print("\nTraining selesai.")
    print("Hasil eksperimen:", results.save_dir)
    print("Bobot terbaik:", Path(results.save_dir) / "weights" / "best.pt")

if __name__ == "__main__":
    main()

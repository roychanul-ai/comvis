from ultralytics import YOLO

# Membangun arsitektur YOLO26n dari konfigurasi,
# tanpa memuat bobot pretrained.
model = YOLO("yolo26n.yaml")

# Melatih model dari inisialisasi acak.
results = model.train(
    data="datasets/indoor/data.yaml",
    epochs=50,
    imgsz=640,
    batch=2,
    device="cpu",
    pretrained=False,
    workers=2,
    project="outputs/training",
    name="yolo26_indoor_scratch"
)
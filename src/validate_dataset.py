
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "indoor"

SPLITS = ["train", "valid", "test"]
NUM_CLASSES = 10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

errors = []
total_images = 0
total_boxes = 0

for split in SPLITS:
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = {p.stem: p for p in label_dir.rglob("*.txt")}

    image_stems = {p.stem for p in images}

    missing_labels = image_stems - labels.keys()
    orphan_labels = labels.keys() - image_stems

    print(f"\n{split.upper()}")
    print(f"Gambar: {len(images)}")
    print(f"File label: {len(labels)}")

    if missing_labels:
        errors.append(
            f"{split}: {len(missing_labels)} gambar tidak memiliki label"
        )

    if orphan_labels:
        errors.append(
            f"{split}: {len(orphan_labels)} label tidak memiliki gambar"
        )

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            errors.append(f"Gambar tidak dapat dibaca: {image_path.name}")
            continue

        label_path = labels.get(image_path.stem)
        if label_path is None:
            continue

        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            errors.append(f"Gagal membaca {label_path.name}: {exc}")
            continue

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) != 5:
                errors.append(
                    f"{label_path.name}:{line_number} harus memiliki 5 nilai"
                )
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(
                    f"{label_path.name}:{line_number} berisi nilai nonangka"
                )
                continue

            if not 0 <= class_id < NUM_CLASSES:
                errors.append(
                    f"{label_path.name}:{line_number} ID kelas {class_id} "
                    f"di luar rentang 0-{NUM_CLASSES - 1}"
                )

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                errors.append(
                    f"{label_path.name}:{line_number} koordinat di luar 0-1"
                )
            elif w <= 0 or h <= 0:
                errors.append(
                    f"{label_path.name}:{line_number} lebar/tinggi harus positif"
                )
            elif x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6:
                errors.append(
                    f"{label_path.name}:{line_number} kotak melewati batas horizontal"
                )
            elif y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
                errors.append(
                    f"{label_path.name}:{line_number} kotak melewati batas vertikal"
                )

            total_boxes += 1

    total_images += len(images)

print("\n=== RINGKASAN ===")
print(f"Total gambar: {total_images}")
print(f"Total anotasi kotak: {total_boxes}")
print(f"Jumlah masalah: {len(errors)}")

for error in errors[:30]:
    print("-", error)

if len(errors) > 30:
    print(f"... dan {len(errors) - 30} masalah lainnya")

if errors:
    print("\nVALIDASI GAGAL. Periksa masalah yang dilaporkan.")
else:
    print("\nVALIDASI DASAR BERHASIL.")

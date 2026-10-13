
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "indoor"

TARGET_PREFIX = "Screenshot-from-2024-04-07-13-31-01_png.rf."

fixed_lines = 0

for label_path in DATASET.rglob("*.txt"):
    if not label_path.name.startswith(TARGET_PREFIX):
        continue

    original_lines = label_path.read_text(encoding="utf-8").splitlines()
    output_lines = []

    for line_number, line in enumerate(original_lines, start=1):
        parts = line.split()

        if len(parts) == 5:
            output_lines.append(line)
            continue

        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            raise ValueError(
                f"Format polygon tidak valid: {label_path.name}, "
                f"baris {line_number}"
            )

        class_id = parts[0]
        coords = list(map(float, parts[1:]))

        xs = coords[0::2]
        ys = coords[1::2]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        output_lines.append(
            f"{class_id} {x_center:.8f} {y_center:.8f} "
            f"{width:.8f} {height:.8f}"
        )
        fixed_lines += 1

    label_path.write_text(
        "\n".join(output_lines) + "\n",
        encoding="utf-8"
    )

print(f"Konversi selesai. Baris polygon yang diubah: {fixed_lines}")

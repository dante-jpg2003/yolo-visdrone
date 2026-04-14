import os
from pathlib import Path
from PIL import Image

# VisDrone class mapping (0-indexed for YOLO)
# Original VisDrone: 0=ignored, 1=pedestrian, ..., 10=motor, 11=others
# We skip class 0 (ignored regions) and remap
VISDRONE_TO_YOLO = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
    # 0 = ignored region, 11 = others — both skipped
}

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

def convert_annotation(ann_path, img_path, out_path):
    img = Image.open(img_path)
    img_w, img_h = img.size

    lines = []
    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            category = int(parts[5])

            if category not in VISDRONE_TO_YOLO:
                continue  # skip ignored/others

            cls = VISDRONE_TO_YOLO[category]

            # Convert to YOLO normalized cx, cy, w, h
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clamp to [0, 1]
            cx = min(max(cx, 0), 1)
            cy = min(max(cy, 0), 1)
            nw = min(max(nw, 0), 1)
            nh = min(max(nh, 0), 1)

            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

def convert_split(visdrone_root, split, output_root):
    """
    split: 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', etc.
    """
    img_dir = Path(visdrone_root) / split / 'images'
    ann_dir = Path(visdrone_root) / split / 'annotations'
    out_img_dir = Path(output_root) / ('train' if 'train' in split else 'val' if 'val' in split else 'test') / 'images'
    out_lbl_dir = Path(output_root) / ('train' if 'train' in split else 'val' if 'val' in split else 'test') / 'labels'

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob('*.jpg'))
    print(f"Converting {len(images)} images for split: {split}")

    for img_path in images:
        ann_path = ann_dir / (img_path.stem + '.txt')
        out_lbl_path = out_lbl_dir / (img_path.stem + '.txt')

        # Symlink or copy image
        out_img_path = out_img_dir / img_path.name
        if not out_img_path.exists():
            os.symlink(img_path.resolve(), out_img_path)

        if ann_path.exists():
            convert_annotation(ann_path, img_path, out_lbl_path)

if __name__ == '__main__':
    VISDRONE_ROOT = 'data/raw'      # where you unzipped VisDrone
    OUTPUT_ROOT = 'data/processed'  # YOLO-format output

    convert_split(VISDRONE_ROOT, 'VisDrone2019-DET-train', OUTPUT_ROOT)
    convert_split(VISDRONE_ROOT, 'VisDrone2019-DET-val', OUTPUT_ROOT)
    convert_split(VISDRONE_ROOT, 'VisDrone2019-DET-test-dev', OUTPUT_ROOT)
    print("Done!")
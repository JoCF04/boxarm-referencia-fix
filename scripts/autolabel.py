from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : Pre-labels new frames with the current model (both classes,
#              brazo and caja_1sol) to speed up manual labeling -- writes
#              .txt files in YOLO format, does not replace human review.
#              Dataset tooling, not part of the pipeline (src/boxarm).
# -----------------------------------------------------------------------
"""Auto-labels images with an existing YOLO model: writes one .txt per
image in YOLO format (class x_center y_center w h, all normalized 0-1)
next to each .jpg, ready to import/fix in Roboflow or LabelImg.

conf=0.1 on purpose: for autolabel it's better to over-detect (false
positives are easy to delete while reviewing) than to under-detect (a
false negative goes unnoticed and silently poisons training, especially
on the occluded boxes that are the current weak spot).

Usage:
    python scripts/autolabel.py
    python scripts/autolabel.py --images data/frames_autolabel --model models/model_br.pt
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("autolabel")

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-labels images with a YOLO model (.txt in YOLO format)")
    ap.add_argument("--images", type=Path, default=ROOT / "data" / "frames_autolabel")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "model_br.pt")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    if not args.model.is_file():
        logger.error("model not found: %s", args.model)
        sys.exit(1)
    imgs = sorted(p for p in args.images.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not imgs:
        logger.error("no images in %s", args.images)
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    logger.info("model classes: %s", model.names)
    logger.info("auto-labeling %d images -- conf=%.2f device=%s", len(imgs), args.conf, args.device)

    total_per_class: dict[str, int] = {}
    no_detections = 0
    for img_path in imgs:
        result = model.predict(
            str(img_path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False,
        )[0]

        lines = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cx, cy, w, h = box.xywhn[0].tolist()  # already normalized 0-1, YOLO format
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            name = model.names[cls_id]
            total_per_class[name] = total_per_class.get(name, 0) + 1

        if not lines:
            no_detections += 1

        dest = img_path.with_suffix(".txt")
        dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # classes.txt next to the images, in the model's index order -- what
    # LabelImg/Roboflow expect to import these .txt files as YOLO labels.
    order = [model.names[i] for i in sorted(model.names)]
    (args.images / "classes.txt").write_text("\n".join(order) + "\n", encoding="utf-8")

    logger.info("done: %s", ", ".join(f"{k}={v}" for k, v in sorted(total_per_class.items())))
    if no_detections:
        logger.warning("%d images with NO detections -- check by hand, could be a hard scene "
                       "(the model's current weak spot) or an empty/invalid frame",
                       no_detections)
    logger.info(".txt files saved next to each image in %s (YOLO format, classes.txt included)", args.images)


if __name__ == "__main__":
    main()

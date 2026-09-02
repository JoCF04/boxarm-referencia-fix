"""Genera la paleta de colores del ISO desde referencias de cada clase.

El color no se predice como una clase adicional: YOLO localiza las cajas y
OpenCV mide el color de una zona interior de cada bbox. Para una imagen con
varias cajas de la misma clase se usa la mediana de todas las muestras.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISION = ROOT / "configs" / "vision.yaml"
DEFAULT_REFS = ROOT / "data" / "ref"
DEFAULT_OUTPUT = ROOT / "configs" / "color_cls.json"


def _class_names(vision_path: Path) -> list[str]:
    with vision_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return [str(name) for name in raw["boxes"]["class_names"]]


def _reference_path(ref_dir: Path, class_name: str) -> Path:
    candidates = sorted(
        path for path in ref_dir.glob(f"{class_name}.*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not candidates:
        raise FileNotFoundError(
            f"No existe referencia para {class_name!r} en {ref_dir}; "
            f"se esperaba {class_name}.jpg/.png"
        )
    return candidates[0]


def _dominant_bgr(crop: np.ndarray) -> np.ndarray:
    """Color BGR predominante del crop, ignorando blancos/negros y sombras.

    Promediar (o mediar) todos los pixeles mezcla el color real de la caja
    con reflejos blancos, sombras y bordes oscuros: un azul saturado termina
    "lavado" hacia celeste, o hacia rojo si domina una etiqueta. En vez de
    eso: se filtra por HSV (fuera blancos de baja saturacion y negros de bajo
    valor) y se agrupan los pixeles restantes por bin de matiz (hue), tomando
    la mediana BGR del bin mas poblado -> el color mas frecuente, no el
    promedio de una mezcla de colores.
    """
    pixels = crop.reshape(-1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    # Descarta blancos (poco saturados) y negros/sombras (poco luminosos).
    colorful = (sat >= 60) & (val >= 40) & (val <= 250)
    candidates = pixels[colorful]
    hue_candidates = hue[colorful]

    if candidates.size == 0:
        # Caja acromatica (blanco/negro/gris real): usa todo el recorte.
        candidates = pixels
        hue_candidates = hue

    # Agrupa por matiz en bins de 10 grados (0-179 en OpenCV) y se queda con
    # el bin mas poblado: el color que realmente domina, no una mezcla.
    bins = (hue_candidates.astype(np.int32) // 10)
    counts = np.bincount(bins, minlength=18)
    dominant_bin = int(np.argmax(counts))
    dominant_mask = bins == dominant_bin
    dominant_pixels = candidates[dominant_mask]
    if dominant_pixels.size == 0:
        dominant_pixels = candidates

    return np.median(dominant_pixels, axis=0)


def _sample_rgb(image: np.ndarray, bbox: tuple[int, int, int, int]) -> list[int]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    # Evita bordes, texto y fondo. El 50% central suele representar mejor el
    # color de la caja que la bbox completa.
    dx = max(1, int((x2 - x1) * 0.25))
    dy = max(1, int((y2 - y1) * 0.25))
    crop = image[
        max(0, y1 + dy):min(height, y2 - dy),
        max(0, x1 + dx):min(width, x2 - dx),
    ]
    if crop.size == 0:
        raise ValueError(f"bbox invalido para muestreo: {bbox}")
    bgr = _dominant_bgr(crop)
    return [int(round(bgr[2])), int(round(bgr[1])), int(round(bgr[0]))]


def _sample_reference(image: np.ndarray) -> list[int]:
    """Mide referencias diagramadas que no contienen detecciones YOLO.

    Algunas plantillas son renders de layout, no fotos etiquetables por el
    detector. Se descarta el fondo oscuro y se toma la mediana de los pixeles
    claros del area central.
    """
    crop = image[
        int(image.shape[0] * 0.08):int(image.shape[0] * 0.92),
        int(image.shape[1] * 0.08):int(image.shape[1] * 0.92),
    ]
    rgb = crop[:, :, ::-1]
    mask = np.max(rgb, axis=2) >= 150
    pixels = rgb[mask]
    if pixels.size == 0:
        pixels = rgb.reshape(-1, 3)
    return [int(round(median(channel))) for channel in pixels.T]


def calibrate_class(model: YOLO, image_path: Path, class_name: str) -> list[int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer la referencia: {image_path}")
    result = model.predict(source=image, verbose=False, conf=0.25)[0]
    names = result.names
    samples: list[list[int]] = []
    for box, cls_id in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        detected_name = str(names[int(cls_id)])
        if detected_name != class_name:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        samples.append(_sample_rgb(image, (x1, y1, x2, y2)))
    if not samples:
        print(f"{class_name}: sin bbox YOLO; usando muestra de referencia diagramada")
        return _sample_reference(image)
    return [int(median(channel)) for channel in zip(*samples)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", help="calibrar solo esta clase")
    parser.add_argument("--image", type=Path, help="referencia puntual para --class")
    parser.add_argument("--vision", type=Path, default=DEFAULT_VISION)
    parser.add_argument("--refs", type=Path, default=DEFAULT_REFS)
    parser.add_argument("--weights", type=Path, help="pesos YOLO de cajas")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    names = _class_names(args.vision)
    selected = [args.class_name] if args.class_name else names
    unknown = sorted(set(selected) - set(names))
    if unknown:
        parser.error(f"clases ausentes en vision.yaml: {', '.join(unknown)}")
    if args.image and len(selected) != 1:
        parser.error("--image requiere --class")

    weights = args.weights
    if weights is None:
        with args.vision.open(encoding="utf-8") as stream:
            weights = ROOT / Path((yaml.safe_load(stream) or {})["boxes"]["weights"])
    model = YOLO(str(weights))

    output = {}
    for class_name in selected:
        reference = args.image if args.image else _reference_path(args.refs, class_name)
        rgb = calibrate_class(model, reference, class_name)
        output[class_name] = {"rgb": rgb, "hex": "#%02x%02x%02x" % tuple(rgb)}
        print(f"{class_name}: RGB={rgb} <- {reference}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"version": 1, "space": "rgb", "classes": output}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"guardado: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

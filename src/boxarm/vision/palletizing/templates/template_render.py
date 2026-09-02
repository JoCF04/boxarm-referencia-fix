from __future__ import annotations

"""Render de depuracion y CLI de plantillas sobre canvas vacio."""

from pathlib import Path

from .template_runtime import (
    LayoutTemplate,
    get_layout_template_pair,
    get_template_box_classes,
)


def render_layout_templates(
    box_class: str,
    output_dir: Path | str,
    base_pattern: str = "auto",
) -> tuple[Path, Path, Path]:
    """Dibuja A, B y una comparacion, sin usar imagen de camara."""
    import cv2
    import numpy as np

    templates = get_layout_template_pair(box_class)
    if templates is None:
        raise ValueError(f"no existen plantillas para {box_class!r}")
    selected = base_pattern.upper()
    if selected not in {"AUTO", "A", "B"}:
        raise ValueError("base_pattern debe ser auto, A o B")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def draw(template: LayoutTemplate, name: str) -> np.ndarray:
        size, margin = 900, 70
        pallet = size - 2 * margin
        canvas = np.full((size, size, 3), (24, 28, 31), dtype=np.uint8)
        cv2.rectangle(
            canvas,
            (margin - 18, margin - 18),
            (size - margin + 18, size - margin + 18),
            (62, 112, 155),
            -1,
        )
        cv2.rectangle(
            canvas,
            (margin, margin),
            (size - margin, size - margin),
            (53, 72, 80),
            -1,
        )
        for slot in template.slots:
            cx = int(margin + slot.u * pallet)
            cy = int(margin + slot.v * pallet)
            width = int(slot.width * pallet)
            height = int(slot.height * pallet)
            p0 = (cx - width // 2, cy - height // 2)
            p1 = (cx + width // 2, cy + height // 2)
            cv2.rectangle(canvas, p0, p1, (220, 228, 232), -1)
            cv2.rectangle(canvas, p0, p1, (20, 165, 235), 4)
            text = f"{slot.cell:02d}{slot.orientation.value}"
            cv2.putText(
                canvas,
                text,
                (cx - 28, cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (25, 35, 40),
                2,
            )
        title = f"{box_class} - PATRON {name} ({len(template.slots)} cajas)"
        cv2.putText(
            canvas,
            title,
            (margin, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (235, 240, 245),
            2,
        )
        return canvas

    pattern_a = draw(templates[0], "A")
    pattern_b = draw(templates[1], "B")
    path_a = output / f"{box_class}_pattern_A.png"
    path_b = output / f"{box_class}_pattern_B.png"
    path_both = output / f"{box_class}_patterns.png"
    cv2.imwrite(str(path_a), pattern_a)
    cv2.imwrite(str(path_b), pattern_b)
    cv2.imwrite(str(path_both), cv2.hconcat((pattern_a, pattern_b)))
    return path_a, path_b, path_both


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Renderiza las dos plantillas fisicas de paletizado.",
    )
    parser.add_argument(
        "--box-class",
        default="all",
        choices=("all",) + get_template_box_classes(),
        help="clase a dibujar o all para todas las clases activas",
    )
    parser.add_argument(
        "--base-pattern",
        default="auto",
        choices=("auto", "A", "B"),
        help="patron activo en N0; auto significa que bootstrap prueba ambos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/layout_templates"),
    )
    args = parser.parse_args()
    classes = get_template_box_classes() if args.box_class == "all" else (args.box_class,)
    for box_class in classes:
        outputs = render_layout_templates(
            box_class,
            args.output_dir,
            args.base_pattern,
        )
        print(f"{box_class}: {len(outputs)} imagenes")
        for path in outputs:
            print(path.resolve())
    return 0


__all__ = ["main", "render_layout_templates"]

"""Calibra el ROI principal y el ROI del pallet como poligonos de n lados.

Flujo: se elige una imagen con un dialogo de tkinter, luego se dibuja el
poligono del ROI principal haciendo click punto por punto sobre un Canvas de
tkinter (con linea de previsualizacion siguiendo el cursor). Si el cursor
esta cerca de un vertice ya puesto (del mismo poligono o del ROI anterior) se
"pega" a el (snap) para reutilizarlo. Click cerca del primer vertice cierra
el poligono solo. Se repite para el ROI del pallet. El resultado se guarda en
configs/roi_cam_<N>.json (una camara por archivo, `--camera N` elige cual)
con los vertices en pixeles absolutos (relativos a la imagen original) y
normalizados (0 a 1, dividiendo por ancho/alto).

Ese JSON es la UNICA fuente de los dos poligonos: el pipeline lee `main_roi`
como el ROI de deteccion de esa camara y `pallet_roi` como las esquinas de
la paleta 3D. Por eso ninguno de los dos se repite ya en pipeline.yaml ni en
drawing.yaml.

Del archivo de imagen se guarda solo el NOMBRE, no la ruta completa: el JSON
se versiona y una ruta absoluta de una maquina no significa nada en otra
(G-6). Es una anotacion de procedencia, nada lo abre.

Controles dentro de cada ventana de poligono:
  - click izquierdo: agregar vertice (o cerrar el poligono si esta cerca del
    primer vertice)
  - click derecho: cancelar el poligono actual y empezar de nuevo
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs"

Point = tuple[int, int]  # x, y en pixeles de la imagen original

SNAP_RADIUS_DISPLAY = 12  # px en pantalla; atrae el cursor a vertices existentes
CLOSE_RADIUS_DISPLAY = 15  # px en pantalla; click cerca del primer punto cierra el poligono
MIN_POINTS = 3


def _pick_image() -> Path:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Elegir imagen para calibrar ROI",
        filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")],
    )
    root.destroy()
    if not path:
        raise SystemExit("No se selecciono ninguna imagen")
    return Path(path)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class PolygonSelector:
    """Ventana tkinter para dibujar un poligono click a click, con snap a
    vertices existentes (los del poligono actual y los de otros ya puestos,
    pasados como `other_points_orig`) y linea de previsualizacion."""

    def __init__(
        self,
        image: Image.Image,
        title: str,
        other_points_orig: list[Point] | None = None,
        max_side: int = 1000,
    ) -> None:
        self._image = image
        self._other_points_orig = other_points_orig or []
        self._scale = min(1.0, max_side / max(image.width, image.height))
        display_size = (
            max(1, int(image.width * self._scale)),
            max(1, int(image.height * self._scale)),
        )
        self._display_image = image.resize(display_size)

        self._root = tk.Tk()
        self._root.title(title)
        self._status = tk.Label(
            self._root,
            text=f"{title} — click: agregar vertice | click derecho: cancelar",
        )
        self._status.pack()

        self._photo = ImageTk.PhotoImage(self._display_image)
        self._canvas = tk.Canvas(
            self._root, width=display_size[0], height=display_size[1], cursor="cross"
        )
        self._canvas.pack()
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

        self._display_points: list[tuple[float, float]] = []
        self._polygon: list[Point] | None = None

        # vertices de otros poligonos, en coordenadas de pantalla, para snap
        self._other_display_points = [self._to_display(p) for p in self._other_points_orig]

        self._canvas.bind("<Motion>", self._on_motion)
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<Button-3>", self._on_cancel_polygon)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close_window)

        self._draw_static()

    def _to_display(self, point: Point) -> tuple[float, float]:
        return (point[0] * self._scale, point[1] * self._scale)

    def _snap_target(self, x: float, y: float) -> tuple[float, float] | None:
        candidates = self._display_points + self._other_display_points
        best = None
        best_dist = SNAP_RADIUS_DISPLAY
        for cand in candidates:
            d = _dist((x, y), cand)
            if d < best_dist:
                best_dist = d
                best = cand
        return best

    def _on_close_window(self) -> None:
        self._polygon = None
        self._root.destroy()

    def _on_cancel_polygon(self, _event: tk.Event) -> None:
        self._display_points = []
        self._status.config(text="Poligono cancelado — empeza de nuevo")
        self._draw_static()

    def _on_motion(self, event: tk.Event) -> None:
        self._draw_static(cursor=(event.x, event.y))

    def _on_click(self, event: tk.Event) -> None:
        x, y = event.x, event.y
        snap = self._snap_target(x, y)
        px, py = snap if snap is not None else (x, y)

        if len(self._display_points) >= MIN_POINTS:
            if _dist((px, py), self._display_points[0]) <= CLOSE_RADIUS_DISPLAY:
                self._finish()
                return

        self._display_points.append((px, py))
        self._draw_static()

    def _draw_static(self, cursor: tuple[float, float] | None = None) -> None:
        self._canvas.delete("overlay")

        for x, y in self._other_display_points:
            self._canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3, outline="cyan", width=1, tags="overlay"
            )

        for x, y in self._display_points:
            self._canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4, fill="lime", outline="lime", tags="overlay"
            )
        if len(self._display_points) > 1:
            flat = [c for point in self._display_points for c in point]
            self._canvas.create_line(*flat, fill="lime", width=2, tags="overlay")

        if cursor is not None:
            cx, cy = cursor
            snap = self._snap_target(cx, cy)
            if snap is not None:
                cx, cy = snap
                self._canvas.create_oval(
                    cx - 8, cy - 8, cx + 8, cy + 8, outline="yellow", width=2, tags="overlay"
                )
            if self._display_points:
                last = self._display_points[-1]
                self._canvas.create_line(last[0], last[1], cx, cy, fill="lime", width=1, tags="overlay")
            if len(self._display_points) >= MIN_POINTS:
                first = self._display_points[0]
                if _dist((cx, cy), first) <= CLOSE_RADIUS_DISPLAY:
                    self._canvas.create_line(
                        cx, cy, first[0], first[1], fill="red", width=2, tags="overlay"
                    )

        self._status.config(
            text=f"{len(self._display_points)} punto(s) — click cerca del primero para cerrar"
        )

    def _finish(self) -> None:
        self._polygon = [
            (int(x / self._scale), int(y / self._scale)) for x, y in self._display_points
        ]
        self._root.destroy()

    def run(self) -> list[Point]:
        self._root.mainloop()
        if not self._polygon or len(self._polygon) < MIN_POINTS:
            raise SystemExit("No se completo un poligono valido (minimo 3 puntos)")
        return self._polygon


def _normalize(points: list[Point], width: int, height: int) -> list[dict[str, float]]:
    return [{"x": x / width, "y": y / height} for x, y in points]


def _absolute(points: list[Point]) -> list[dict[str, int]]:
    return [{"x": x, "y": y} for x, y in points]


def _camera_id(value: int | None) -> int:
    """Obtiene la cámara explícita, sin asumir silenciosamente la 1."""
    if value is None:
        try:
            value = int(input("¿Qué cámara quieres calibrar? (1, 2, 3): ").strip())
        except (EOFError, ValueError) as exc:
            raise SystemExit("Debes indicar un número de cámara válido") from exc
    if value < 1:
        raise SystemExit("--camera debe ser >= 1 (es la posicion en camaras[] de pipeline.yaml)")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="id de camara (/cam/<id>): decide el archivo configs/roi_cam_<id>.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="ruta explicita de salida; por defecto configs/roi_cam_<camera>.json",
    )
    args = parser.parse_args()

    camera = _camera_id(args.camera)
    output = args.output or CONFIGS_DIR / f"roi_cam_{camera}.json"

    image_path = _pick_image()
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    print("Elige el ROI principal: click punto por punto, cierra clickeando cerca del primero")
    main_roi = PolygonSelector(image, "Elige el ROI principal").run()
    print(f"ROI principal completado con {len(main_roi)} puntos")

    print("Elige el ROI del pallet: click punto por punto, cierra clickeando cerca del primero")
    pallet_roi = PolygonSelector(image, "Elige el ROI del pallet", other_points_orig=main_roi).run()
    print(f"ROI del pallet completado con {len(pallet_roi)} puntos")

    data = {
        # solo el nombre: ver docstring del modulo
        "image": image_path.name,
        "image_size": {"width": width, "height": height},
        "main_roi": {
            "absolute": _absolute(main_roi),
            "normalized": _normalize(main_roi, width, height),
        },
        "pallet_roi": {
            "absolute": _absolute(pallet_roi),
            "normalized": _normalize(pallet_roi, width, height),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ROI de la camara {camera} guardado en {output}")


if __name__ == "__main__":
    main()

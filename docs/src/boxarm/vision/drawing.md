# src/boxarm/vision/drawing.py

Anotaciones sobre el frame. No decide nada de deteccion ni de conteo,
solo pinta sobre un frame ya procesado por
[inference.py](inference.md), con los `GridDetection` que resuelve
[palletizing.py](palletizing.md). Colores y layout vienen de
`DrawingConfig` ([configs/drawing.md](../../../configs/drawing.md)), no
hay constantes hardcodeadas en el modulo.

## Interfaz publica

- `color_for_level(level, cfg, brightness=1.0)` — color BGR estable por
  nivel, compartido por el overlay 2D y el
  [ISO](isometric.md). `brightness` da variantes del MISMO tono (1.0 caja
  nueva, mas oscuro re-deteccion) sin confundir nivel con estado.
- `draw_roi(frame, roi_pts, cfg)` — contorno de la zona de conteo.
- `draw_grid_detections(frame, results, cfg)` — una caja por
  `GridDetection`: las aceptadas con el color de su nivel y la etiqueta
  `"7N1"` (mismo formato que el ISO, para cruzar las dos vistas), solo la
  NUEVA se marca aparte; las rechazadas en rojo con el codigo del motivo.
- `draw_arm_present(frame, roi_pts, cfg)` — aviso "BRAZO" cuando el brazo
  esta en escena (sin dibujar detecciones de cajas).
- `draw_hud(frame, tag, counter, inf_fps, cfg)` — panel con tag de
  camara, total en paleta (desglosado en inicial + brazo), visibles y FPS
  de inferencia. `counter` solo necesita cumplir el protocolo minimo
  `total` / `visible` (y opcionalmente `initial` / `placed`).

`cfg` es siempre un `DrawingConfig`.

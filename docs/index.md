# CNM-Robotic_Box_Arm — Documentacion

Sistema de vision artificial para el conteo y verificacion de cajas en
pallets usando 3 camaras simultaneas (Jetson Orin AGX en produccion, o
videos de prueba en cualquier PC). Cada camara corre en su propio
proceso; un unico servidor Flask sirve los 3 streams por ruta
(`/cam/1`, `/cam/2`, `/cam/3`).

## Contenido

- [main.py](main.md) — entry point, sin argumentos
- [docker.md](docker.md) — correr en la Jetson dentro del contenedor
  (permisos de `/dev/video*`, puertos)
- [src/boxarm/](src/boxarm/index.md) — paquete con toda la logica,
  organizado en subpaquetes `vision/`, `capture/`, `runtime/`, `web/`
- [configs/pipeline.yaml](configs/pipeline.md) — captura, streaming, camaras
- [configs/vision.yaml](configs/vision.md) ? detecci?n YOLO frame a frame, sin tracker
- [configs/drawing.yaml](configs/drawing.md) — colores y layout de las anotaciones
- [configs/palletizing.yaml](configs/palletizing.md) — conteo por rejilla y
  escala aparente ([docs/palletizing_counting.md](palletizing_counting.md)),
  con estado espacial por celda/nivel
- [Prompt visual del visor ISO](iso_design_prompt.md) — dirección de diseño industrial minimalista
- [known_issues.md](known_issues.md) — deuda de tests preexistente, sin resolver a propósito

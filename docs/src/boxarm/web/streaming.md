# src/boxarm/web/streaming.py

Streaming HTTP: un unico Flask, un unico puerto, sirviendo todas las
camaras por ruta.

## Interfaz publica

- `drain_jpeg_queue(cam_id, jpeg_q, runtime, stop)` — hilo en el proceso
  principal, uno por camara: drena la `multiprocessing.Queue` de esa
  camara hacia el buffer compartido `_latest_jpeg[cam_id]`, con timeout
  `runtime.drain_timeout_s`.
- `make_flask_app(cameras, runtime) -> Flask` — una sola app con rutas:
  - `/cam/<id>` — stream MJPEG (`multipart/x-mixed-replace`) de la
    camara con ese `id` (posicion 1-based en `configs/pipeline.yaml`),
    con poll `runtime.mjpeg_poll_s` mientras no hay frame.
  - `/` — pagina indice con un link por camara.

`runtime` es un `RuntimeConfig` (`PipelineConfig.runtime`, ver
[configs/pipeline.md](../../../configs/pipeline.md)). Reemplaza la
version anterior de N apps Flask con N puertos.

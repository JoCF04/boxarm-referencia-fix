# configs/pipeline.yaml

Captura, streaming y cámaras. Cargado con `yaml.safe_load` y validado
una sola vez en `boxarm.config.load_pipeline_config()`.

Detecci?n vive en [vision.md](vision.md); colores/layout de dibujado en
[drawing.md](drawing.md); host/puerto/calidad JPEG del servidor Flask en
[web.md](web.md) — sin claves repetidas entre los archivos.

## Claves

| Clave | Tipo | Descripcion |
| ----- | ---- | ----------- |
| `modo` | str | `"camera"` (Jetson, GStreamer+NVDEC) o `"video"` (archivos de prueba) |
| `loop_video` | bool | repetir el video de prueba al llegar al final (solo `modo: video`) |
| `device` | str | dispositivo de inferencia YOLO (`"cuda"` en la Jetson) |
| `fps_request` | int | FPS solicitados a cada camara (solo `modo: camera`) |
| `cap_width` / `cap_height` | int | resolucion de captura solicitada |
| `weights` | ruta | pesos YOLO, relativa a la raiz del repo, compartidos por las 3 camaras |
| `runtime.reconnect_delay_s` | float | espera antes de reintentar abrir una camara caida |
| `runtime.read_fail_delay_s` | float | espera antes de reintentar tras un fallo de lectura |
| `runtime.gst_timeout_s` | float | timeout de `try_pull_sample` en el appsink de GStreamer |
| `runtime.mjpeg_poll_s` | float | intervalo de poll del generador MJPEG mientras no hay frame |
| `runtime.drain_timeout_s` | float | timeout de lectura de la cola JPEG entre procesos |
| `runtime.shutdown_join_timeout_s` | float | timeout al unir hilos/procesos en el apagado |
| `runtime.main_loop_tick_s` | float | intervalo del loop principal esperando Ctrl+C |
| `camaras[].index` | int | numero de `/dev/videoN` (solo `modo: camera`) |
| `camaras[].tag` | str | nombre para logs y para la pagina indice de Flask |
| `camaras[].video` | ruta | video de prueba (solo `modo: video`), relativa a la raiz del repo |
| `camaras[].enabled` | bool (opcional, default `true`) | `false` -> esa camara no arranca proceso ni ruta `/cam/<id>` (util para probar con menos camaras de las que hay configuradas) |

## ROI y esquinas de la paleta

Ninguno de los dos se declara en este archivo. Cada camara los toma de
`configs/roi_cam_<id>.json`, donde `<id>` es el mismo de `/cam/<id>` (la
posicion 1-based en `camaras`):

| Campo del JSON | Uso |
| --- | --- |
| `main_roi.normalized` | ROI de deteccion (4 vertices `{x, y}` en [0,1] del frame): decide si un bbox esta dentro del area vigilada. Se escala a pixeles al arrancar contra el tamano real del primer frame. A proposito mas amplio que la paleta, para no recortar cajas altas |
| `pallet_roi.normalized` | esquinas reales de la tarima. Es la entrada de la homografia del conteo (`GridCounter`): lleva la superficie de la paleta al cuadrado unidad `[0,1]^2` donde viven celdas, footprints y solapes. Tambien dibuja el deck en `/cam/<id>/iso` |

Los dos poligonos tienen trabajos distintos y no son intercambiables: el
`main_roi` filtra, el `pallet_roi` rectifica. Las distancias de
`configs/palletizing.yaml` (`tau_cell`, `partial_fit_tolerance`,
`max_position_correction`) estan en unidades de ese cuadrado unidad, o sea
en fracciones de paleta -- si se recalibra el `pallet_roi` con otro tamano,
hay que reescalarlas en el mismo factor.

El archivo lo escribe `python scripts/calibrate_roi.py --camera <id>`. Si
falta el JSON de una camara, el arranque falla con un `ConfigError` que dice
que comando correr -- no hay ROI por defecto.

El `<id>` de la ruta `/cam/<id>` **no** se configura: es la posicion
1-based de cada camara en la lista `camaras` (la primera es `/cam/1`,
la segunda `/cam/2`, etc.) y no cambia si otra camara tiene
`enabled: false` -- el numerado no se corre.

## Ejemplo minimo

```yaml
modo: video
loop_video: true
device: cuda
weights: models/model_br.pt
runtime:
  reconnect_delay_s: 2
  read_fail_delay_s: 1
  gst_timeout_s: 2.0
  mjpeg_poll_s: 0.02
  drain_timeout_s: 0.5
  shutdown_join_timeout_s: 3
  main_loop_tick_s: 1.0
camaras:
  - index: 0
    tag: "Camara 1"    # -> /cam/1
    video: videos/p2r1.mp4
```

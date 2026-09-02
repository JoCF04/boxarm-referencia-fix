# configs/web.yaml

Todo lo que es servidor/transporte HTTP: host y puerto del unico Flask que
sirve el dashboard y cada `/cam/<id>`, y la calidad JPEG del stream MJPEG.
Nada de captura ni deteccion aca -- eso vive en [pipeline.md](pipeline.md).

Cargado con `yaml.safe_load` y validado en `boxarm.config.load_web_config()`.
`load_pipeline_config()` lo busca solo como hermano de `pipeline.yaml`
dentro de `configs/` y lo expone en `PipelineConfig.web`.

## Claves

| Clave | Tipo | Descripcion |
| ----- | ---- | ----------- |
| `flask_host` | str | host de bind del unico servidor Flask (`"0.0.0.0"` = todas las interfaces) |
| `port` | int | unico puerto Flask -- cada camara se sirve en `/cam/<id>` |
| `jpeg_quality` | int | calidad JPEG del stream MJPEG (0-100); OpenCV clampea fuera de rango |
| `stream_max_fps` | float | maximo de FPS transmitidos por cliente MJPEG; no limita captura, inferencia ni conteo |

## Ejemplo minimo

```yaml
flask_host: "0.0.0.0"
port: 8080
jpeg_quality: 55
stream_max_fps: 8.0
```

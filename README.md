# CNM-Robotic_Box_Arm

Sistema de visión para detectar y contar cajas paletizadas con varias cámaras,
YOLO y un visor web 2D/3D.

## Arquitectura

```text
main.py
  └─ boxarm.runtime.pipeline.run()
      ├─ proceso por cámara
      │   ├─ capture.camera_io      captura
      │   ├─ vision.inference      detección y clasificación YOLO
      │   ├─ vision.palletizing    estado espacial (celda, nivel)
      │   ├─ vision.drawing        overlay 2D
      │   └─ vision.isometric      escena visual 3D
      └─ web.streaming             Flask + MJPEG + escena ISO
```

La identidad operativa de cada caja se determina por su clase de producto y
su posición persistente en la paleta: `(clase, celda, nivel)`. El backend
detecta y clasifica las cajas, mientras el módulo de palletizado consolida su
posición usando la ocupación espacial y las plantillas de la clase activa.

El contrato técnico del conteo está en
[`docs/palletizing_counting.md`](docs/palletizing_counting.md).
La extracción y organización de plantillas está documentada en
[`docs/src/templates_extraction.md`](docs/src/templates_extraction.md).

## Configuración

Cada archivo tiene una responsabilidad única:

| Archivo | Responsabilidad |
|---|---|
| `configs/pipeline.yaml` | Cámaras, captura, ROI por cámara, runtime y grabación |
| `configs/web.yaml` | Host/puerto y calidad JPEG del servidor Flask (streaming/dashboard) |
| `configs/vision.yaml` | Modelos YOLO, clases, confianza, resolución e inferencia |
| `configs/palletizing.yaml` | Calibración, umbrales, ocupación y reglas del conteo |
| `configs/isometric.yaml` | Geometría de la paleta y parámetros de la vista ISO |
| `configs/drawing.yaml` | Colores y estilos de las anotaciones 2D |

La configuración activa usa `layout_mode: auto`. La capacidad por nivel no se
configura en YAML: se obtiene de las plantillas de cada clase en
`src/boxarm/vision/palletizing/templates/box/`.

Hay 12 clases previstas y cada una tiene su módulo de plantilla. Las clases
con plantillas extraídas y activas se registran en
`templates/template_runtime.py`; las restantes están preparadas como módulos
para incorporar sus patrones posteriormente.

Para regenerar las imágenes de las plantillas:

```powershell
$env:PYTHONPATH = "src"
python src/boxarm/vision/palletizing/layout_templates.py --box-class all
```

Los resultados se guardan en `data/layout_templates/`.

## Regla robot–ROI

El conteo se pausa cuando el bbox del robot intersecta el ROI de la paleta.
Durante esa intersección:

- se ignoran las detecciones de cajas para el estado;
- no se pintan cajas observadas;
- se muestra el bbox del robot;
- no se modifica la posición, el tamaño ni el nivel de cajas confirmadas.

## Ejecución

Requisitos principales: Python 3.10+, PyTorch, Ultralytics, OpenCV, Flask,
PyYAML y NumPy.

```powershell
$env:PYTHONPATH = "src"
python main.py
```

`main.py` carga los seis archivos YAML desde `configs/` (`web.yaml` se
carga como hermano de `pipeline.yaml`, sin llamada propia en `main.py`).

La interfaz web escucha en `8080` por defecto. El transporte continúa
separado por rutas: `/cam/<id>/stream` y `/cam/<id>/iso/scene`.

## Endpoints

| Ruta | Descripción |
|---|---|
| `/cam/<id>` | Panel unificado: video + gemelo digital 3D |
| `/cam/<id>/stream` | Stream MJPEG anotado |
| `/cam/<id>/snapshot` | Ultimo frame anotado (JPEG estático) |
| `/cam/<id>/iso/scene` | Escena JSON para el renderer Canvas |
| `/api/cameras` | Estado y última escena de todas las cámaras |
| `/` | Dashboard multi-cámara (resumen, cámaras e inspección) |

La página ISO rota y hace zoom localmente con `requestAnimationFrame`; no
solicita un JPEG nuevo al servidor por cada movimiento del usuario.

## Conteo

El contador mantiene estas garantías:

1. Una ocupación confirmada solo cambia de `0 → 1`.
2. Un nivel confirmado nunca disminuye.
3. El nivel y el tamaño confirmados son inmutables; la posición solo admite
   una corrección estable y no ambigua en modo `auto`.
4. Después del inventario inicial se cuenta una caja por ciclo del brazo.
5. Una caja de `z+1` requiere que el nivel `z` esté completo.
6. Una caja superior requiere al menos dos soportes inferiores cruzados.

## Pruebas

```powershell
$env:PYTHONPATH = "src"
pytest -q
node --test tests/test_iso_js.mjs
```

## Estructura principal

```text
configs/
data/layout_templates/
docs/
src/boxarm/
  capture/
  runtime/
  vision/
    palletizing/
      templates/box/
  web/
tests/
main.py
```

# configs/vision.yaml

Configuración de detección YOLO frame a frame. Se carga mediante
`boxarm.config.load_vision_config()` como `VisionConfig`.

```yaml
detection:
  conf: 0.01
  imgsz: 416
  arm_class_name: cnm-palletsCajas
  box_class_names:
    - caja_cartucho_0.10
    - caja_cartucho_1.00
    - caja_cartucho_2.00
  class_conf:
    caja_cartucho_0.10: 0.45
    caja_cartucho_1.00: 0.50
    caja_cartucho_2.00: 0.45
    cnm-palletsCajas: 0.08

fps_smoothing_alpha: 0.85
subsample_factor: 2
```

## Clases por nombre, nunca por id

`models/model_br.pt` expone hoy:

| id | nombre | rol |
|---|---|---|
| 0 | `caja_cartucho_0.10` | caja contable |
| 1 | `caja_cartucho_1.00` | caja contable |
| 2 | `caja_cartucho_2.00` | caja contable |
| 3 | `cnm-palletsCajas` | **brazo** |

Los ids salen del orden del `names:` del dataset y **se corren al agregar
etiquetas nuevas**; los nombres no. Por eso `arm_class_name`,
`box_class_names` y las claves de `class_conf` se escriben con el nombre de
la clase, y `_model_class_ids()` los resuelve contra `model.names` al
arrancar.

Configurar por id fue un bug real y costoso: `arm_class_id: 2` se escribio
cuando el id 2 era el brazo, pero tras reentrenar el id 2 paso a ser
`caja_cartucho_2.00`. La validacion de entonces solo comprobaba que el id
existiera — y existia — asi que no habia error. El efecto: cada caja de dos
soles se trataba como brazo, pausaba el conteo, cerraba ciclos de brazo
fantasma, y las cajas de esa denominacion nunca se contaban.

Con nombres, una clase ausente aborta el arranque con el listado real del
modelo. Al terminar de etiquetar las denominaciones (0.10/0.20/0.50/1/2/5) y
presentaciones (bolsa / cartucho) que faltan, basta agregar el nombre nuevo a
`box_class_names` — ningun id se toca.

`arm_class_name` no puede repetirse dentro de `box_class_names`, y
`box_class_names` no puede quedar vacio. Ambas condiciones se validan al
cargar el yaml. Las claves `arm_class_id` / `box_class_ids` fueron eliminadas:
si aparecen en el yaml, la carga falla indicando el reemplazo.

Al arrancar, cada camara deja registrado que resolvio:

```
[Camara 3] clases resueltas -- brazo: 'cnm-palletsCajas'=3; cajas: 'caja_cartucho_0.10'=0, ...
```

| Clave | Descripción |
|---|---|
| `detection.conf` | confianza mínima de una detección |
| `detection.imgsz` | resolución de entrada de YOLO |
| `fps_smoothing_alpha` | suavizado de la métrica FPS |
| `subsample_factor` | procesa uno de cada N frames capturados |

No existe `tracker_yaml`, `track_id`, `counting_mode` ni configuración de
ByteTrack. La identidad persistente es `(celda, nivel)` y se configura en
[palletizing.md](palletizing.md).

`min_stable` y `same_box_iou` también están en `palletizing.yaml`: describen
cuándo confirmar una caja candidata, no cómo ejecuta YOLO la detección.

`detection.conf` es el piso global enviado a YOLO y debe ser el menor umbral
requerido. `detection.class_conf` aplica después el mínimo por clase. Solo las
clases de `box_class_names` llegan al contador; las demás se ignoran.

El umbral del brazo se deja deliberadamente bajo (`0.08`): un falso positivo
de brazo solo pausa el conteo, mientras que un falso negativo deja contar con
el brazo tapando cajas. Los umbrales de caja son altos porque un fantasma sí
se convierte en una identidad persistente.

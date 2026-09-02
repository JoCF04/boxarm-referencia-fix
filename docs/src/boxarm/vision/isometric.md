# src/boxarm/vision/isometric.py

Implementacion de la seccion 10 de
[docs/palletizing_counting.md](../../../palletizing_counting.md): vista
de inspeccion 3D (axonometria, orden de pintado por profundidad, culling
de caras traseras). Solo diagnostico visual, no afecta el conteo. Se
sirve en `/cam/<id>/iso`.

**Renderizador puro.** El modulo no toma ninguna decision: no mide
footprints, no deduce niveles, no calcula alturas ni aplica umbrales.
Todo eso lo resuelve el cerebro ([palletizing.py](palletizing.md)) y
llega ya resuelto en un `SceneState`.

## Interfaz publica

- `render_isometric(scene: SceneState, drawing_cfg: DrawingConfig, iso: IsometricConfig, azimuth_deg=None, elevation_deg=None) -> np.ndarray`
  — dibuja `scene.boxes` como paralelepipedos, ordenadas por profundidad
  (lejos -> cerca, algoritmo del pintor) y solo con las caras cuya normal
  apunta hacia la camara (culling por producto punto, sin z-buffer,
  valido por convexidad del prisma). `azimuth_deg`/`elevation_deg`
  permiten girar la camara sin tocar el yaml (arrastre con el mouse en el
  visor web); si no se pasan, se usan los de `iso`.

`iso` trae la geometria real del render
([configs/isometric.md](../../../configs/isometric.md)): dimensiones del
pallet en metros, angulos de vista, tamano del canvas, `fill_margin` y
`visual_height_ratio`.

## Que aporta el SceneState

| Campo | Uso en el render |
| ----- | ---------------- |
| `boxes[].cell`, `.level` | color de cara (por celda) y de arista (por nivel), etiqueta `"7N1"` |
| `boxes[].u`, `.v` | centro en el cuadrado unidad del pallet |
| `boxes[].side_a`, `.side_b` | footprint YA unificado al consenso del nivel |
| `boxes[].z0`, `.height` | base y extrusion visual de la caja |
| `overlaps[]` | prisma de interseccion (`u0,v0,u1,v1`, `z0`, `height`) y `ratio` para la etiqueta |
| `level_tops` | marcas del eje Z (cotas acumuladas, `len == levels + 1`) |
| `total_height` | altura de la torre para centrar y auto-escalar la escena |
| `total`, `initial`, `placed`, `levels` | texto del HUD del ISO |

## Escalas

El `SceneState` vive en el cuadrado unidad del pallet. El render lo lleva
a metros con `pallet_width_m` en X, `pallet_length_m` en Y y
`min(pallet_width_m, pallet_length_m)` en Z (el eje Z no tiene un lado
propio: la extrusion es una proporcion visual, no una medicion).

La escala px/m se AUTO-CALCULA para que la escena completa (pallet en X,Y
y la torre entera en Z) llene `fill_margin` del canvas.

## Sobre el footprint unificado

Todas las cajas del pallet son la misma caja fisica, pero su footprint en
el cuadrado unidad NO es igual en todos los niveles: la homografia esta
calibrada sobre el piso (z=0), asi que una caja mas alta esta mas cerca
de la camara y se proyecta MAS GRANDE. Esa diferencia es justamente la
senal con la que se deduce el nivel: unificar el footprint ENTRE niveles
la borraria y toda la pila se dibujaria como una sola capa.

Dentro de un nivel, en cambio, toda diferencia de tamano es error de
medicion (un bbox medido bajo oclusion queda congelado). Se corrige con
la MEDIANA del lado largo y del corto de ese nivel — robusta frente a una
minoria de bbox recortados o inflados — conservando la orientacion propia
de cada caja, para que una caja girada siga viendose girada.

Ese calculo ya NO vive en este modulo: lo hace `palletizing.py` al
construir `SceneBox.side_a/side_b`. Se documenta aqui porque es el
razonamiento que explica lo que se ve en pantalla.

## Formulas (seccion 10.A del documento)

```text
x' = X cos(theta) - Y sin(theta)
y' = X sin(theta) + Y cos(theta)
u  = x'
v  = -(y' sin(phi) + z cos(phi))
d  = -y' cos(phi) + z sin(phi)      -- profundidad, lejano primero
```

# configs/drawing.yaml

Colores (BGR, no RGB) y layout de las anotaciones sobre el frame: ROI,
cajas detectadas y HUD. Cargado con `yaml.safe_load` y validado una sola
vez en `boxarm.config.load_drawing_config()`.

Captura/streaming vive en [pipeline.md](pipeline.md);
detecci?n en [vision.md](vision.md) — sin claves repetidas
entre los tres archivos.

## Claves

| Clave | Descripcion |
| ----- | ----------- |
| `colors.new` | verde -> caja nueva confirmada |
| `colors.redet` | amarillo -> re-deteccion de caja ya contada |
| `colors.pending` | naranja -> aun no estable |
| `colors.roi` | color del contorno de la zona de conteo |
| `colors.text` | color de texto generico del HUD |
| `colors.hud_title` | color del tag de camara en el HUD |
| `colors.arm_alert` | color del aviso "BRAZO" |
| `roi.thickness` | grosor del contorno de la ROI |
| `box.thickness` | grosor del rectangulo de cada deteccion |
| `box.label_font_scale` / `box.label_thickness` | texto "ID N estado" sobre cada caja |
| `box.circle_radius` | punto en el centro de cada caja |
| `arm_alert.font_scale` / `arm_alert.thickness` | texto "BRAZO" |
| `hud.width` / `hud.height` | tamano del panel del HUD |
| `hud.background` | color de fondo del panel |
| `hud.title_font_scale` / `hud.title_thickness` | linea con el tag de camara |
| `hud.line_font_scale` / `hud.line_thickness` | linea "Colocadas" |
| `hud.visible_thickness` | grosor de la linea "Visibles" (mismo `line_font_scale`) |
| `hud.fps_font_scale` / `hud.fps_thickness` | linea "FPS infer" |

## Ejemplo minimo

```yaml
colors:
  new:       [0, 255, 0]
  redet:     [255, 200, 0]
  pending:   [0, 165, 255]
  roi:       [255, 200, 0]
  text:      [255, 255, 255]
  hud_title: [0, 210, 255]
  arm_alert: [0, 80, 255]

roi: { thickness: 2 }
box: { thickness: 2, label_font_scale: 0.55, label_thickness: 2, circle_radius: 4 }
arm_alert: { font_scale: 0.7, thickness: 2 }
hud:
  width: 280
  height: 150
  background: [20, 20, 20]
  title_font_scale: 0.75
  title_thickness: 2
  line_font_scale: 0.85
  line_thickness: 2
  visible_thickness: 1
  fps_font_scale: 0.75
  fps_thickness: 1
```

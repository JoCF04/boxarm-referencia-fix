# configs/palletizing.yaml

Reglas f?sicas y calibraci?n del contador espacial. El gate vive aqu? porque
decidir si un frame es v?lido es l?gica de paletizado, no de inferencia.

## Claves principales

- `layout_mode`: `auto` descubre posiciones; `fixed` exige centros calibrados.
- `boxes_per_level`: capacidad f?sica por clase. La decision de si el
  arranque necesita resolver una composicion `N0/N1` NO se basa en comparar
  el conteo de cajas visibles contra este numero (una caja apilada puede
  hacer que el conteo total coincida con la capacidad igual) -- se basa en
  si existe al menos un footprint parcial/ocluido (ver
  `palletizing_counting.md` seccion 18.0).
- `levels_layout`: centros `(u,v)` por nivel para modo fijo.
- `occupancy_grid`: resoluci?n del mapa usado para estimar cobertura/huecos.
- `max_bootstrap_combinations`: tope al producto cartesiano de hipotesis del
  solver inicial. Por encima de ese numero el frame se deja "sin resolver" y
  se reintenta en el siguiente, en vez de enumerar millones de combinaciones.
- `gate.motion_pause_enabled`: pausa adicional por movimiento.
- `gate.motion_diff_threshold`: umbral del cambio dentro del ROI.
- `gate.motion_stable_frames`: frames quietos para reanudar.
- `gate.arm_debounce_frames`: frames sin robot para cerrar un ciclo.
- `gate.empty_pallet_debounce_frames`: frames `COUNTING` consecutivos sin
  ninguna detección, habiendo cajas ya confirmadas, antes de asumir que la
  paleta se vació en la realidad y resetear el conteo entero
  (`GridCounter.reset_pallet()`) para la próxima carga. Es la única
  transición 1→0 de `chi(g,z)` de todo el paquete -- ver la nota de
  monotonicidad en `palletizing_math.md` §4, que documenta la invariante
  *dentro* de una paleta, no a través de un reemplazo de paleta.
- `confirmation.min_stable`: frames `COUNTING` consecutivos para confirmar
  cualquier hipotesis geometrica, incluidas todas las fronteras `Ni -> N(i+1)` del arranque.
- `confirmation.same_box_iou`: IoU mínimo para reconocer esa candidata durante la confirmación corta.
- `thresholds.tau_cell_overlap`: matching espacial con ocupaci?n confirmada.
- `thresholds.min_stack_area_ratio`: rechaza fragmentos demasiado peque?os.
- `thresholds.min_support_coverage`: cobertura conjunta m?nima de los soportes.
- `thresholds.max_support_ratio`: dominio relativo máximo `s1/s2` entre los
  dos soportes principales; evita confundir coincidencia 1-a-1 con amarre.

La asociación por IoU termina al confirmar. La identidad persistente es
`(celda,nivel)`: no hay `track_id`, y footprints/niveles confirmados no se
refinan con observaciones posteriores.

En modo `auto`, el nivel persistente nunca se renumera. El seguimiento usa una
ventana local deslizante: al completar `Ni`, ese nivel pasa a local 0,
`N(i+1)` pasa a local 1 y los niveles absolutos inferiores dejan de participar
en matching sin borrarse del JSON ni del ISO.

No existe un `max_levels` configurable ni un tope operativo arbitrario. El
contador avanza `N0, N1, N2, ...` mientras la calibracion proyectiva sea
fisicamente valida. El unico dominio finito se deduce de `c_z` y `box_height`:
la cara superior del nivel debe permanecer por debajo del centro optico,
`n * box_height < c_z`. Si una instalacion alcanza ese dominio, se debe
recalibrar la geometria; poner un numero grande como 100 ocultaria una
calibracion imposible y produciria escalas negativas o divergentes.

La configuraci?n actual declara 15 cajas por nivel, pero los layouts de muestra
tienen 10 posiciones repetidas. Mantener `layout_mode: auto` hasta calibrar los
patrones A/B reales.

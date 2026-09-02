from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : Se unifican en una sola puerta (_resolve_level) los cuatro
#              mecanismos que decidian el nivel y que hasta ahora competian
#              repartidos por update(): escalera s(z), override por
#              oclusion, apilamiento geometrico y filtro de gravedad. La
#              decision pasa a viajar como dato (LevelSource en
#              GridDetection) en vez de existir solo en el log.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : Este modulo pasa a ser el UNICO que decide. Absorbe el gate
#              (pausa por brazo/movimiento) y el ciclo de brazo, que vivian
#              en el lazo de inferencia, y expone scene_state() con la carga
#              ya resuelta para que el render isometrico no calcule nada.
#              Los umbrales dejan de estar hardcodeados: viven en
#              configs/palletizing.yaml (regla G-5). Renombrado desde
#              grid_counting.py -- el nombre ahora dice el dominio.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- Changelog
# Programmer  | Date     | Resumen
# ----------- | -------- | -----------------------------------------------
# gerald      | 23-08-26 | Conteo por identidad de celda (g, z) y escala
#             |          | aparente, en vez de seguimiento temporal por ID
# gerald      | 23-08-26 | Celdas por posicion declarada (patron irregular
#             |          | tipo rompecabezas), no rejilla filas x columnas
# gerald      | 25-08-26 | Se parte el modulo unico (palletizing.py, 3000+
#             |          | lineas) en el paquete vision/palletizing/, con
#             |          | GridCounter compuesto por mixins de un dominio
#             |          | cada uno (init/estado, niveles, bootstrap,
#             |          | matching, lazo de frame, escena). Ningun nombre
#             |          | ni comportamiento publico cambio.
# -----------------------------------------------------------------------
"""GridCounter: el unico cerebro del pipeline de paletizado.

Cuenta cajas por celda (g) y nivel (z) sobre una paleta vista desde una
camara cenital, sin seguimiento temporal. En modo ``auto`` las celdas se
descubren desde las detecciones; en modo ``fixed`` se asignan contra
LevelLayout. Ambos admiten arreglos irregulares.

Regla de arquitectura del proyecto: TODA decision vive aca. El lazo de
inferencia (`inference.py`) solo observa y dibuja; el render isometrico
(`isometric.py`) solo pinta lo que `scene_state()` le entrega ya resuelto.
Si algo hay que decidir sobre la carga -- si vale la pena mirar el frame,
en que nivel esta una caja, de que tamano se dibuja -- se decide en este
paquete y en ningun otro. Y ningun umbral vive hardcodeado aqui: todos
salen de configs/palletizing.yaml (regla G-5).

Superficie publica:

- ``update(FrameInput) -> FrameResult``  -- unica entrada por frame
- ``scene_state(height_ratio) -> SceneState`` -- unica salida para el render
- ``set_box_class(str)`` -- que denominacion se esta paletizando

Resumen del metodo (ver docs/palletizing_counting.md para la derivacion
completa):

- Los 4 vertices de la paleta (`CameraConfig.roi`, calibrado en
  configs/roi_cam_<id>.json) definen una homografia `H`
  que rectifica la vista a un cuadrado unidad [0,1]^2 (seccion 4).
- Cada nivel `z` tiene una scale aparente esperada `s(z)`, estrictamente
  creciente porque una caja mas alta esta mas cerca de la camara
  (seccion 5.A). El signo del error separa oclusion (recorta, `E<s`) de
  nivel superior (amplia, `E>s`) -- Proposicion 5.1.
- La identidad de una caja es el par `(celda, nivel)` que ocupa, no su
  historia temporal (Proposicion 8.1). Aqui "celda" es el indice de la
  posicion declarada mas cercana en LevelLayout.cells, no una casilla de
  rejilla. El estado es una funcion de ocupacion `chi(g,z)` que solo
  transiciona 0->1 por construccion: este paquete nunca escribe False
  sobre una celda ya True, asi que la invariante de la Observacion 8.2
  (ninguna transicion 1->0) se cumple por diseno, no por chequeo en
  tiempo de ejecucion.
- El conteo `n = sum(chi)` es monotono no decreciente (Proposicion 9.1).

Lo que este paquete NO implementa (fuera de alcance por ahora, ver
docs/palletizing_counting.md secciones 10 y F4): reconstruccion 3D de
inspeccion, y la verificacion de recalibracion de vertices por ciclo.

Organizacion interna (P-7: un modulo = una responsabilidad; GridCounter
es la composicion de todos los mixins, sin logica propia):

- ``types``       -- enums y dataclasses publicas del dominio
- ``formulas``     -- funciones puras de geometria/numeros, sin estado
- ``init_state``   -- construccion y persistencia (state_dict/save/load)
- ``levels``       -- decision de nivel y celda, footprint y soporte
- ``bootstrap``    -- reconciliacion del inventario inicial Ni/N(i+1)
- ``matching``     -- confirmacion temporal, deduplicacion, matching a celdas
- ``frame_loop``   -- puerta de entrada `update()`, gate, conteo por frame
- ``scene``        -- `scene_state()` y verificacion de cierre
"""

from .bootstrap import _BootstrapMixin
from .frame_loop import _FrameLoopMixin
from .init_state import _InitStateMixin
from .levels import _LevelsMixin
from .matching import _MatchingMixin
from .scene import _SceneMixin


class GridCounter(
    _InitStateMixin,
    _LevelsMixin,
    _BootstrapMixin,
    _MatchingMixin,
    _FrameLoopMixin,
    _SceneMixin,
):
    """Conteo de cajas por celda/nivel para UNA camara/paleta: el unico
    contador del pipeline."""

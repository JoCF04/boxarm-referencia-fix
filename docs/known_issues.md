# Known issues — deuda de tests preexistente

Encontrado auditando la suite completa al preparar el fix de reset de
paleta vacía (ver `configs/palletizing.md` → `gate.empty_pallet_debounce_frames`
y `palletizing_math.md` §4). Ninguno de estos puntos tiene que ver con ese
fix ni fue tocado por él — quedan documentados acá para que alguien del
equipo los levante como tarea aparte, con la cita exacta de dónde falla
cada uno.

Reproducir con (falta instalar `pytest`, `scipy` y opcionalmente `rich`
en el entorno de desarrollo — no están en `docker/requirements_container.txt`
como dependencias de test):

```
pip install pytest scipy rich
PYTHONPATH=src python -m pytest tests/ -q
```

## 1. `PalletizingConfig` no tiene `boxes_per_level` — 13+ tests lo asumen

**Síntoma:** `AttributeError: 'PalletizingConfig' object has no attribute
'boxes_per_level'`.

**Dónde:**

- `tests/test_bootstrap_performance.py:36` (`_template_counter()`, 5 tests
  la usan)
- `tests/test_template_bootstrap_partials.py::test_fourteen_complete_plus_two_fragments_apply_only_after_min_stable`
  (falla distinto — ver punto 4 — pero viene de la misma raíz)
- 13 tests en `tests/test_palletizing.py::AutoLayoutTests` que arman
  `cfg.boxes_per_level = {...}` a mano sobre el fixture

**Causa probable:** la capacidad por nivel migró de un campo de config
(`boxes_per_level`) a `get_template_capacity(box_class)` leyendo
`vision/palletizing/templates/` (ver comentarios en `init_state.py` y
`frame_loop.py` que ya referencian `get_template_capacity`), pero los
tests que dependían del modelo viejo no se actualizaron. `PalletizingConfig`
(`src/boxarm/config.py`) tampoco declara ese campo — así que no es
solamente que el test viejo quedó atrás, el propio dataclass ya no lo
tiene.

**No lo resolví porque:** requiere decidir si `boxes_per_level` se
reintroduce en `PalletizingConfig` (y de dónde sale ese valor ahora) o si
estos 13+ tests se reescriben contra el modelo de templates — decisión de
diseño que no me corresponde.

## 2. `test_template_bootstrap_partials.py` cuenta 29 en vez de 15

**Síntoma:** `assert counter.total == 15` falla con `29 == 15`, en
`test_fourteen_complete_plus_two_fragments_apply_only_after_min_stable`
(`tests/test_template_bootstrap_partials.py:99`).

Aparece junto con warnings de calibración de escalera
(`la escalera no separa los niveles...`) que sugieren que el `cfg` de
prueba que arma este archivo (`_counter()`, línea 16, con el mismo
`base.boxes_per_level` del punto 1) ya no calza con `get_template_capacity`
para `coin_roll_100`. Sin resolver el punto 1 no tiene sentido investigar
este número por separado.

## 3. `test_logging_config.py` — depende de `rich`, que es opcional

**Síntoma:** `TypeError: isinstance() arg 2 must be a type, a tuple of
types, or a union`, en los dos tests de
`tests/test_logging_config.py` (`test_configure_logging_uses_rich_for_an_interactive_terminal`,
`test_configure_logging_falls_back_when_stream_is_not_interactive`).

**Causa:** `src/boxarm/runtime/logging_config.py:12-17` importa `rich` en
un `try/except ImportError` y deja `RichHandler = None` si no está
instalado (es una mejora opcional, no un requisito — correcto para
producción). Los tests hacen `isinstance(handler, logging_config.RichHandler)`
sin contemplar que `RichHandler` puede ser `None` en un entorno sin
`rich` instalado — deberían saltarse (`pytest.mark.skipif`) en vez de
fallar.

**No lo resolví porque:** es un fix de una línea (el `skipif`) pero
implica decidir si `rich` pasa a ser una dependencia de test declarada en
algún requirements de desarrollo que hoy no existe, o si el skip es la
política deseada.

## 4. `test_recording.py::test_pipeline_exposes_three_recording_types`

**Síntoma:** `recording.type_enabled("normal") is True` falla — da
`False`.

**Causa:** el test carga `configs/pipeline.yaml` real y asume
`recording.enabled: true`, pero el archivo real tiene
`recording.enabled: false` (línea `enabled: false` bajo `recording:`).
`type_enabled()` combina el interruptor global con el del tipo (ver
`RecordingConfig.type_enabled` en `src/boxarm/config.py`), así que con el
global en `false` cualquier tipo da `False` aunque `types.normal: true`.

**No lo resolví porque:** no sé si `recording.enabled: false` en el
YAML real es intencional (grabación desactivada a propósito en este
checkout) o quedó así por error — cambiarlo afecta comportamiento de
producción, no solo el test.

## 5. Dos archivos de test no cargan (error de colección, no de aserción)

- `tests/test_inference_arm_roi.py:10` — `ImportError: cannot import name
  '_active_box_class' from 'boxarm.vision.inference'`. Esa función ya no
  existe en `src/boxarm/vision/inference.py`.
- `tests/test_layout_templates.py:19` — `ModuleNotFoundError: No module
  named 'boxarm.vision.palletizing.templates.coin_roll_100'`. Ese módulo
  de template no existe en `src/boxarm/vision/palletizing/templates/` en
  este checkout.

Ambos impiden que pytest siquiera recolecte esos archivos (`pytest tests/`
sin más corta con "Interrupted: 2 errors during collection" hasta que se
los excluye a mano con `--ignore`).

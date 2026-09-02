# src/boxarm/runtime/workers.py

Entry point de `multiprocessing.Process` por camara.

## Interfaz publica

- `camera_worker(cam, cfg, vision_cfg, drawing_cfg, jpeg_q, stop)` —
  corre en su propio proceso; arranca un hilo lector
  (`camera_io.reader`, con `vision_cfg.subsample_factor`) y un hilo de
  inferencia (`inference.run_inference`, con `vision_cfg` y `drawing_cfg`)
  dentro de ese proceso. `jpeg_q` es una `multiprocessing.Queue` que
  cruza el JPEG anotado hacia el proceso principal (ver
  [../web/streaming.md](../web/streaming.md)); `stop` es un
  `multiprocessing.Event` compartido por todas las camaras.

Si el proceso de una camara muere, las demas camaras no se ven
afectadas — el proceso principal solo pierde ese stream.

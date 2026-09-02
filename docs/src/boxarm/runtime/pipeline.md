# src/boxarm/runtime/pipeline.py

Orquestacion de todo el sistema.

## Interfaz publica

- `run(cfg, vision_cfg, drawing_cfg)` — filtra `cfg.cameras` a las que
  tienen `enabled: true` (ver
  [configs/pipeline.md](../../../configs/pipeline.md)) y por cada una de
  esas:
  1. crea una `multiprocessing.Queue` (`jpeg_q`)
  2. arranca un `multiprocessing.Process` (`workers.camera_worker`, con
     `cfg`, `vision_cfg` y `drawing_cfg`)
  3. arranca un hilo `drain-<tag>` (`streaming.drain_jpeg_queue`)

  Luego arranca el unico Flask (`streaming.make_flask_app`) en un hilo,
  y espera Ctrl-C en el hilo principal (`cfg.runtime.main_loop_tick_s`).
  Al recibir la senal: setea el `multiprocessing.Event` compartido, une
  los hilos drain y los procesos de camara con timeout
  (`cfg.runtime.shutdown_join_timeout_s`) y los termina a la fuerza si
  no respondieron (P-16 — apagado limpio).

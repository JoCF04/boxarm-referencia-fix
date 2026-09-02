# src/boxarm/vision/inference.py

Bucle de inferencia YOLO frame a frame. Ejecuta `model.predict()` y no conserva
IDs temporales.

## Interfaz

`run_inference(cfg, vision_cfg, drawing_cfg, palletizing_cfg, isometric_cfg,
cam, frame_q, jpeg_out, iso_jpeg_out, scene_out, iso_view, stop)` consume frames,
separa robot/cajas, calcula movimiento y entrega un `FrameInput` a
`GridCounter.update()`.

`arm_visible` solo es verdadero cuando el bbox del robot intersecta el ROI. Si
el gate bloquea, no se entregan ni pintan cajas. La vista web recibe el ?ltimo
`SceneState` por una cola latest-only; el Canvas rota localmente.

La configuraci?n de YOLO viene de
[`configs/vision.yaml`](../../../configs/vision.md). Las reglas f?sicas viven en
[`configs/palletizing.yaml`](../../../configs/palletizing.md).

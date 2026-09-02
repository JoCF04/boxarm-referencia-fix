# src/boxarm/capture/camera_io.py

Captura de frames. No sabe nada de YOLO ni de conteo, solo entrega
frames a una `queue.Queue` local.

## Interfaz publica

- `open_camera(cam_idx, tag, cap_width, cap_height, fps_request, gst_timeout_s)`
  — abre `/dev/video{cam_idx}` probando, en orden: GStreamer+NVDEC (MJPEG
  con decodificacion por hardware), GStreamer YUYV crudo, y V4L2 plano
  via `cv2.VideoCapture`. `gst_timeout_s` viene de
  `PipelineConfig.runtime.gst_timeout_s`.
- `GstCamera` — wrapper GStreamer con la misma interfaz minima que
  `cv2.VideoCapture` (`read()`/`release()`).
- `push_frame(frame_q, frame)` — empuja descartando el mas viejo si la
  cola esta llena (prioriza el frame mas reciente).
- `reader(cam, cfg, subsample_factor, frame_q, stop)` — despacha a
  captura en vivo o video de prueba segun `cfg.modo`. `subsample_factor`
  viene de `VisionConfig.subsample_factor`
  ([configs/vision.md](../../../configs/vision.md)). `stop` acepta
  `threading.Event` o `multiprocessing.Event` (misma interfaz
  `is_set/set/wait`).

Reconexion automatica (`cfg.runtime.reconnect_delay_s` /
`read_fail_delay_s`) si la camara falla, sin matar el hilo.

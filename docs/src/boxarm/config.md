# src/boxarm/config.py

Configuraci?n tipada cargada desde YAML.

- `VisionConfig`: `DetectionConfig`, suavizado FPS y submuestreo; sin tracker.
- `PalletizingConfig`: gate, capacidad, layouts, niveles y soportes.
- `DrawingConfig`: anotaciones y colores.
- `IsometricConfig`: presentaci?n 3D.
- `PipelineConfig`: c?maras, captura, runtime, streaming y grabaci?n.

Loaders p?blicos: `load_pipeline_config`, `load_vision_config`,
`load_drawing_config`, `load_palletizing_config` y `load_isometric_config`.

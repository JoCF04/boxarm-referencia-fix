# main.py

Entry point del pipeline. Sin logica propia: carga
`configs/pipeline.yaml` y llama a `boxarm.pipeline.run(cfg)`. Toda la
implementacion vive en [src/boxarm/](src/boxarm/index.md).

## Interfaz publica

```bash
python main.py
```

Sin argumentos. La configuracion viene de tres YAML de ruta fija:
[configs/pipeline.md](configs/pipeline.md) (captura/streaming),
[configs/vision.md](configs/vision.md) (detecci?n frame a frame) y
[configs/drawing.md](configs/drawing.md) (colores/layout).

## Por que src/ en sys.path

El repo no tiene todavia un `pyproject.toml` que instale `boxarm` como
paquete editable, asi que `main.py` agrega `src/` a `sys.path` antes de
importar — layout `src/` (P-7) sin paso de instalacion previo.

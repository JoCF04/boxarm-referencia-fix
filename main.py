from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : Entry point (P-8): sin argumentos, sin logica propia --
#              toda la implementacion vive en src/boxarm/ (config.py +
#              subpaquetes vision/, capture/, runtime/, web/).
#              Orquestador raiz del repo (G-4).
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : Carga los YAML de configuracion por separado -- pipeline
#              (captura/streaming), vision (deteccion frame-a-frame) y
#              drawing (colores/layout) -- en vez de mezclar todo en uno.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : Carga tambien palletizing.yaml (conteo por rejilla) e
#              isometric.yaml (vista de inspeccion 3D, /cam/<id>/iso) --
#              5 YAML en total, cada uno un concern separado.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 30-08-26
# Reason     : flask_host/port/jpeg_quality salen de pipeline.yaml a su
#              propio configs/web.yaml (no se mezcla "como se sirve" con
#              "como se captura y detecta"). load_pipeline_config() lo
#              carga solo, como hermano de pipeline.yaml -- no hay llamada
#              nueva aca en main.py.
# -----------------------------------------------------------------------
"""Entry point del pipeline de deteccion y conteo de cajas para N camaras.

Sin argumentos: la configuracion viene de YAML en configs/ --
pipeline.yaml (captura/streaming, con web.yaml como hermano para host/
puerto/calidad JPEG del servidor), vision.yaml (deteccion sin tracker),
drawing.yaml (colores/layout), palletizing.yaml (conteo por rejilla,
ISO) e isometric.yaml (vista de inspeccion 3D,
/cam/<id>). Ver docs/configs/. Ejecutar con:

    python main.py
"""

import logging
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))  # layout src/ (P-7), sin paso de instalacion

from boxarm.config import (  # noqa: E402
    ConfigError,
    load_drawing_config,
    load_isometric_config,
    load_palletizing_config,
    load_pipeline_config,
    load_vision_config,
)
from boxarm.runtime.pipeline import run  # noqa: E402
from boxarm.runtime.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG_PATH             = CONFIGS_DIR / "pipeline.yaml"
DEFAULT_VISION_CONFIG_PATH      = CONFIGS_DIR / "vision.yaml"
DEFAULT_DRAWING_CONFIG_PATH     = CONFIGS_DIR / "drawing.yaml"
DEFAULT_PALLETIZING_CONFIG_PATH = CONFIGS_DIR / "palletizing.yaml"
DEFAULT_ISOMETRIC_CONFIG_PATH   = CONFIGS_DIR / "isometric.yaml"


def main() -> int:
    # BOXARM_LOG_LEVEL=DEBUG para ver el detalle del gate (pausa/reanuda
    # conteo por movimiento o brazo) y de la reconciliacion inicial de
    # cada camara -- MUY verboso para dejarlo prendido siempre, por eso es
    # opt-in por variable de entorno y no un nivel fijo. Sin variable de
    # entorno no cambia nada del comportamiento actual (default INFO). No
    # es un argumento de linea de comandos (P-8: main.py no toma args).
    log_level_name = os.environ.get("BOXARM_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    configure_logging(level=log_level)
    # El visor consulta /iso/scene periodicamente. Un access-log INFO por
    # consulta no aporta diagnostico y oculta los eventos del paletizado;
    # se conservan warnings y errores HTTP.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    try:
        cfg             = load_pipeline_config(DEFAULT_CONFIG_PATH)
        vision_cfg      = load_vision_config(DEFAULT_VISION_CONFIG_PATH)
        drawing_cfg     = load_drawing_config(DEFAULT_DRAWING_CONFIG_PATH)
        palletizing_cfg = load_palletizing_config(DEFAULT_PALLETIZING_CONFIG_PATH)
        isometric_cfg   = load_isometric_config(DEFAULT_ISOMETRIC_CONFIG_PATH)
    except ConfigError as exc:
        logger.critical("configuracion invalida: %s", exc)
        return 1

    run(cfg, vision_cfg, drawing_cfg, palletizing_cfg, isometric_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : Claude
# Date       : 02-09-26
# Reason     : Historial de produccion para reportes ("cuantas paletas por
#              dia", "de que denominacion", "cuantas cajas tuvo cada una").
#              GridCounter ya sabia detectar el cierre de una paleta (fix
#              de reset de pallet vacio, ver frame_loop.py); esto solo deja
#              constancia de ese evento en un archivo aparte, sin tocar
#              nada del conteo.
# -----------------------------------------------------------------------
"""Registro append-only de paletas completadas, en SQLite (stdlib puro).

No es el estado operativo de GridCounter (ese sigue siendo el JSON por
camara en `PalletizingConfig.state_directory`, que se sobreescribe en cada
paleta). Esto es historia: una fila por paleta que se dio por completada,
pensada para consultarse despues con `consultar_historial.py`.

Un solo archivo para las 3 camaras -- son pocas escrituras (una por paleta
cerrada, no por frame), asi que el lock de SQLite entre los procesos de
camara no es un cuello de botella real."""

import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Mismo patron que BOXARM_WEB_ENABLED/BOXARM_STALL_DUMP (runtime/pipeline.py,
# runtime/workers.py): env var opcional, default sin sorpresas.
DEFAULT_DB_PATH = Path("historial_paletas.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paletas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha_hora TEXT NOT NULL,
    denominacion TEXT,
    total_cajas INTEGER NOT NULL,
    camara TEXT
)
"""


def db_path() -> Path:
    override = os.environ.get("BOXARM_HISTORY_DB")
    return Path(override) if override else DEFAULT_DB_PATH


def record_pallet_completion(
    denominacion: str | None,
    total_cajas: int,
    camara: str,
    fecha_hora: datetime | None = None,
    path: Path | None = None,
) -> None:
    """Deja constancia de UNA paleta que se acaba de dar por completada.

    Se llama desde `GridCounter._count_boxes()` justo antes de
    `reset_pallet()`, mientras `self.total`/`self._box_class` todavia
    describen la paleta que se retiro (reset_pallet los vuelve a cero).

    Abre y cierra la conexion en cada llamada: son pocas escrituras por
    dia, no vale la pena mantener una conexion viva por proceso de camara.
    Un fallo de disco/lock no puede tumbar el conteo -- es una escritura a
    un archivo externo, un limite del sistema como cualquier I/O, asi que
    se loguea y se sigue en vez de propagar la excepcion."""
    target = path or db_path()
    when = fecha_hora or datetime.now()
    try:
        conn = sqlite3.connect(target, timeout=10)
        try:
            conn.execute(_SCHEMA)
            conn.execute(
                "INSERT INTO paletas (fecha_hora, denominacion, total_cajas, camara) "
                "VALUES (?, ?, ?, ?)",
                (
                    when.isoformat(sep=" ", timespec="seconds"),
                    denominacion,
                    total_cajas,
                    camara,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error:
        logger.exception("No se pudo registrar la paleta completada en %s", target)

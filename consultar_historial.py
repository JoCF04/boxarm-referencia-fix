from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : Claude
# Date       : 02-09-26
# Reason     : Consulta de texto plano sobre historial_paletas.db (ver
#              src/boxarm/history.py) -- paletas por dia, por denominacion
#              y promedio de cajas por paleta. Sin dependencias nuevas ni
#              interfaz: se corre y se lee.
# -----------------------------------------------------------------------
"""Uso: python consultar_historial.py [ruta_a_la_base.db]

Sin argumentos usa historial_paletas.db en el directorio actual (mismo
default que src/boxarm/history.py, salvo que BOXARM_HISTORY_DB diga otra
cosa)."""

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))  # layout src/ (P-7), sin paso de instalacion -- ver main.py

from boxarm.history import db_path  # noqa: E402


def _fetch_rows(conn: sqlite3.Connection, sql: str) -> list[tuple]:
    return conn.execute(sql).fetchall()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else db_path()

    if not path.exists():
        print(f"No existe {path} todavia -- no se registro ninguna paleta.")
        return

    conn = sqlite3.connect(path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM paletas").fetchone()[0]
        print(f"Historial: {path} ({total} paleta(s) registrada(s))\n")

        if total == 0:
            return

        print("Paletas por dia")
        print("-" * 40)
        for fecha, cantidad in _fetch_rows(
            conn,
            "SELECT date(fecha_hora), COUNT(*) FROM paletas "
            "GROUP BY date(fecha_hora) ORDER BY date(fecha_hora)",
        ):
            print(f"  {fecha}: {cantidad} paleta(s)")

        print("\nPaletas por denominacion")
        print("-" * 40)
        for denominacion, cantidad in _fetch_rows(
            conn,
            "SELECT COALESCE(denominacion, '(sin denominacion)') AS d, COUNT(*) AS cantidad "
            "FROM paletas GROUP BY d ORDER BY cantidad DESC",
        ):
            print(f"  {denominacion}: {cantidad} paleta(s)")

        print("\nCajas por paleta")
        print("-" * 40)
        promedio, minimo, maximo = conn.execute(
            "SELECT AVG(total_cajas), MIN(total_cajas), MAX(total_cajas) FROM paletas"
        ).fetchone()
        print(f"  promedio: {promedio:.1f}  minimo: {minimo}  maximo: {maximo}")

        print("\nPor camara")
        print("-" * 40)
        for camara, cantidad in _fetch_rows(
            conn,
            "SELECT COALESCE(camara, '(sin camara)') AS c, COUNT(*) AS cantidad "
            "FROM paletas GROUP BY c ORDER BY cantidad DESC",
        ):
            print(f"  {camara}: {cantidad} paleta(s)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

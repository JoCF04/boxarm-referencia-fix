from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : Claude
# Date       : 02-09-26
# Reason     : Cubre src/boxarm/history.py (registro SQLite de paletas
#              completadas) y su enganche en GridCounter._count_boxes():
#              el reset de paleta vacia (EmptyPalletResetTests en
#              test_palletizing.py) ahora tambien deja una fila en el
#              historial ANTES de poner total/box_class en cero.
# -----------------------------------------------------------------------

import os
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from boxarm.history import record_pallet_completion
from boxarm.vision.palletizing import GridCounter

from test_palletizing import PalletizingTestCase, _gate, _still


class RecordPalletCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / "historial_paletas.db"

    def _rows(self) -> list[tuple]:
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute(
                "SELECT fecha_hora, denominacion, total_cajas, camara FROM paletas"
            ).fetchall()
        finally:
            conn.close()

    def test_writes_one_row_with_the_given_fields(self) -> None:
        record_pallet_completion("coin_roll_100", 42, "Camara 2", path=self.db)

        rows = self._rows()
        self.assertEqual(1, len(rows))
        fecha_hora, denominacion, total_cajas, camara = rows[0]
        self.assertTrue(fecha_hora)  # se genero un timestamp, no vacio
        self.assertEqual("coin_roll_100", denominacion)
        self.assertEqual(42, total_cajas)
        self.assertEqual("Camara 2", camara)

    def test_creates_the_database_file_on_first_write(self) -> None:
        self.assertFalse(self.db.exists())
        record_pallet_completion("bag_10", 5, "Camara 1", path=self.db)
        self.assertTrue(self.db.exists())

    def test_multiple_pallets_accumulate_as_separate_rows(self) -> None:
        record_pallet_completion("coin_roll_10", 10, "Camara 1", path=self.db)
        record_pallet_completion("bag_50", 7, "Camara 3", path=self.db)

        rows = self._rows()
        self.assertEqual(2, len(rows))
        self.assertEqual(["coin_roll_10", "bag_50"], [r[1] for r in rows])

    def test_a_pallet_with_no_resolved_denomination_is_still_recorded(self) -> None:
        record_pallet_completion(None, 3, "Camara 1", path=self.db)

        rows = self._rows()
        self.assertEqual(1, len(rows))
        self.assertIsNone(rows[0][1])
        self.assertEqual(3, rows[0][2])


class GridCounterResetWritesHistoryTests(PalletizingTestCase):
    """El reset de paleta vacia (EmptyPalletResetTests en test_palletizing.py)
    es exactamente el punto donde se cierra una paleta: confirma que ese
    mismo evento deja una fila en el historial antes de poner el conteo
    en cero."""

    def setUp(self) -> None:
        super().setUp()
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / "historial_paletas.db"
        self._previous_env = os.environ.get("BOXARM_HISTORY_DB")
        os.environ["BOXARM_HISTORY_DB"] = str(self.db)
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        if self._previous_env is None:
            os.environ.pop("BOXARM_HISTORY_DB", None)
        else:
            os.environ["BOXARM_HISTORY_DB"] = self._previous_env

    def _counter(self, empty_pallet_debounce_frames: int = 2) -> GridCounter:
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.gate = _gate(empty_pallet_debounce_frames=empty_pallet_debounce_frames)
        return GridCounter(self.roi, cfg, cam_tag="Camara 2")

    def _rows(self) -> list[tuple]:
        if not self.db.exists():  # ninguna paleta se cerro todavia
            return []
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute(
                "SELECT denominacion, total_cajas, camara FROM paletas"
            ).fetchall()
        finally:
            conn.close()

    def test_the_empty_pallet_reset_records_the_finished_pallet(self) -> None:
        counter = self._counter(empty_pallet_debounce_frames=2)
        counter.set_box_class("coin_roll_100")
        counter.update(_still([(10, 40, 30, 60, 0.95)]))
        self.assertEqual(1, counter.total)

        for _ in range(2):  # dispara el reset
            counter.update(_still([]))

        self.assertEqual(0, counter.total)  # el reset SI corrio
        rows = self._rows()
        self.assertEqual(1, len(rows))
        denominacion, total_cajas, camara = rows[0]
        self.assertEqual("coin_roll_100", denominacion)
        self.assertEqual(1, total_cajas)  # el total ANTES del reset, no 0
        self.assertEqual("Camara 2", camara)

    def test_a_reset_below_the_debounce_threshold_records_nothing(self) -> None:
        counter = self._counter(empty_pallet_debounce_frames=3)
        counter.update(_still([(10, 40, 30, 60, 0.95)]))

        for _ in range(2):  # uno menos que el umbral: no dispara
            counter.update(_still([]))

        self.assertEqual([], self._rows())


if __name__ == "__main__":
    unittest.main()

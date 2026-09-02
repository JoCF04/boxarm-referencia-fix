from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : Grabacion opcional a disco de los streams (normal + iso),
#              controlada por configs/pipeline.yaml: recording.enabled --
#              antes no habia forma de guardar el resultado, solo de
#              verlo en vivo por Flask.
# -----------------------------------------------------------------------
"""VideoRecorder: envoltorio de cv2.VideoWriter que se inicializa solo
cuando llega el primer frame (para tomar su tamano real), y que vive en
output_dir/cam/<id>/<kind>_<timestamp>.<extension>."""

import logging
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from boxarm.config import RecordingConfig

logger = logging.getLogger(__name__)

# Se resuelve una sola vez por proceso: shutil.which() ya recorre PATH
# entero, no hace falta repetirlo por cada VideoRecorder que se cierra. None
# = "no buscado todavia" (distinto de "buscado y no esta", que es "").
_ffmpeg_cache: str | None = None
_ffmpeg_searched = False


def _ffmpeg_path() -> str | None:
    """Ruta de ffmpeg en PATH, o None si no esta instalado. Cacheado."""
    global _ffmpeg_cache, _ffmpeg_searched
    if not _ffmpeg_searched:
        _ffmpeg_cache = shutil.which("ffmpeg")
        _ffmpeg_searched = True
        if _ffmpeg_cache is None:
            logger.warning(
                "ffmpeg no esta en PATH -- recording.transcode_h264 queda sin "
                "efecto, los .mp4 grabados quedan en mp4v (WhatsApp y la "
                "mayoria de apps de chat no los abren, solo reproductores "
                "de escritorio)."
            )
    return _ffmpeg_cache


def h264_command(ffmpeg: str, input_path: Path, output_path: Path) -> list[str]:
    """Comando MP4/H.264 interoperable con reproductores moviles y chats."""
    return [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(input_path),
        "-map", "0:v:0", "-an",
        # yuv420p exige dimensiones pares; el recorte ISO puede quedar impar
        # dependiendo del viewport y del redondeo CSS.
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-profile:v", "main", "-level:v", "4.0",
        "-pix_fmt", "yuv420p", "-tag:v", "avc1",
        "-movflags", "+faststart", "-preset", "veryfast", "-crf", "23",
        str(output_path),
    ]


class VideoRecorder:
    """Un archivo por camara y tipo; respeta `recording.types`."""

    def __init__(self, cfg: RecordingConfig, cam_id: int, kind: str, tag: str) -> None:
        self._cfg = cfg
        self._enabled = cfg.type_enabled(kind)
        self._writer: cv2.VideoWriter | None = None
        self._path: Path | None = None
        self._tag = tag
        self._wrote_any = False
        if not self._enabled:
            return

        cam_dir = cfg.output_dir / "cam" / str(cam_id)
        cam_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._path = cam_dir / f"{kind}_{timestamp}.{cfg.extension}"
        logger.info("[%s] grabando %s en %s", tag, kind, self._path)

    def write(self, frame: np.ndarray) -> None:
        if not self._enabled or self._path is None:
            return
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self._cfg.fourcc)
            self._writer = cv2.VideoWriter(str(self._path), fourcc, self._cfg.fps, (w, h))
        self._writer.write(frame)
        self._wrote_any = True

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._cfg.transcode_h264 and self._wrote_any and self._path is not None:
            self._transcode_to_h264()

    def _transcode_to_h264(self) -> None:
        """Reencodea self._path a H.264/yuv420p/faststart con ffmpeg, en el
        mismo archivo. mp4v (lo que escribe cv2.VideoWriter) no lo abren
        WhatsApp ni la mayoria de apps de chat -- piden H.264 especificamente.

        `-movflags +faststart` mueve el indice (moov atom) al principio del
        archivo: sin eso, algunas apps de chat necesitan el archivo completo
        antes de poder previsualizarlo. `-pix_fmt yuv420p` es el formato de
        color que TODOS los reproductores moviles decodifican -- ffmpeg por
        default a veces elige yuv444p/yuv422p segun el origen, que no abre
        en varios telefonos."""
        ffmpeg = _ffmpeg_path()
        if ffmpeg is None:
            return  # ya se logueo el warning una vez en _ffmpeg_path()

        assert self._path is not None
        tmp_path = self._path.with_suffix(".h264.tmp" + self._path.suffix)
        cmd = h264_command(ffmpeg, self._path, tmp_path)
        logger.info("[%s] convirtiendo %s a H.264 para WhatsApp...", self._tag, self._path)
        try:
            subprocess.run(
                cmd, check=True, capture_output=True,
                timeout=self._cfg.transcode_timeout_s,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("[%s] transcode a H.264 fallo para %s (%s) -- se deja el mp4v original",
                           self._tag, self._path, exc)
            tmp_path.unlink(missing_ok=True)
            return

        tmp_path.replace(self._path)
        logger.info("[%s] %s reencodeado a H.264 (compatible con WhatsApp)", self._tag, self._path)

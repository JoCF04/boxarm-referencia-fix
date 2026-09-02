"""Configuracion central del logging de consola del proceso."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

STANDARD_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
RICH_FORMAT = "%(name)s: %(message)s"

try:
    from rich.console import Console
    from rich.logging import RichHandler
except ImportError:  # Rich es una mejora opcional, no un requisito de arranque.
    Console = None  # type: ignore[assignment,misc]
    RichHandler = None  # type: ignore[assignment,misc]


def _is_interactive_terminal(stream: TextIO) -> bool:
    """Indica si Rich puede emitir control de terminal de forma segura."""
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def _make_rich_handler(stream: TextIO):
    if RichHandler is None or Console is None or not _is_interactive_terminal(stream):
        return None

    try:
        console = Console(
            file=stream,
            stderr=True,
            force_terminal=True,
            color_system="auto",
        )
        return RichHandler(
            console=console,
            show_time=True,
            omit_repeated_times=False,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=False,
        )
    except Exception:
        # El logging nunca debe impedir que una camara arranque por una
        # incompatibilidad de terminal o de una version opcional de Rich.
        return None


def configure_logging(level: int = logging.INFO) -> None:
    """Configura una sola vez el root logger del proceso actual.

    En Windows ``multiprocessing`` usa ``spawn``: cada worker entra con un
    interprete nuevo, por lo que debe invocar esta funcion por separado.
    ``force=True`` reemplaza handlers heredados o instalados por librerias y
    evita lineas duplicadas al reconfigurar.
    """
    stream = sys.stderr
    rich_handler = _make_rich_handler(stream)

    if rich_handler is not None:
        logging.basicConfig(
            level=level,
            format=RICH_FORMAT,
            datefmt="[%X]",
            handlers=[rich_handler],
            force=True,
        )
        return

    logging.basicConfig(
        level=level,
        format=STANDARD_FORMAT,
        stream=stream,
        force=True,
    )

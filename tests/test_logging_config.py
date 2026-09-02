from __future__ import annotations

import io
import logging

import pytest

from boxarm.runtime import logging_config


class _TerminalStream(io.StringIO):
    def isatty(self) -> bool:
        return True


class _PlainStream(io.StringIO):
    def isatty(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _restore_root_logging():
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    root.handlers[:] = original_handlers
    root.setLevel(original_level)


def test_configure_logging_uses_rich_for_an_interactive_terminal(monkeypatch):
    stream = _TerminalStream()
    monkeypatch.setattr(logging_config.sys, "stderr", stream)

    logging_config.configure_logging(logging.DEBUG)

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert isinstance(handler, logging_config.RichHandler)
    assert handler.rich_tracebacks is True
    assert handler.formatter is not None
    assert "%(name)s" in handler.formatter._fmt

    logging.getLogger("boxarm.prueba").warning("mensaje visible")
    rendered = stream.getvalue()
    assert "boxarm.prueba" in rendered
    assert "mensaje visible" in rendered
    assert "WARNING" in rendered
    assert handler.console.is_terminal is True


def test_configure_logging_falls_back_when_stream_is_not_interactive(monkeypatch):
    monkeypatch.setattr(logging_config.sys, "stderr", _PlainStream())

    logging_config.configure_logging()

    root = logging.getLogger()
    assert root.level == logging.INFO
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)
    assert not isinstance(root.handlers[0], logging_config.RichHandler)
    assert root.handlers[0].formatter._fmt == logging_config.STANDARD_FORMAT


def test_configure_logging_falls_back_when_rich_is_unavailable(monkeypatch):
    monkeypatch.setattr(logging_config.sys, "stderr", _TerminalStream())
    monkeypatch.setattr(logging_config, "RichHandler", None)

    logging_config.configure_logging(logging.WARNING)

    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1
    assert root.handlers[0].formatter._fmt == logging_config.STANDARD_FORMAT


def test_reconfiguration_does_not_duplicate_handlers(monkeypatch):
    monkeypatch.setattr(logging_config.sys, "stderr", _PlainStream())

    logging_config.configure_logging()
    logging_config.configure_logging(logging.ERROR)

    root = logging.getLogger()
    assert root.level == logging.ERROR
    assert len(root.handlers) == 1

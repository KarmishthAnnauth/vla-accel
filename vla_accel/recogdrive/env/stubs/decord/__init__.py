"""Placeholder for ``decord``, which has no aarch64 wheel.

navsim's internvl_preprocess imports it at module level but image inference never calls it.
"""

__version__ = "0.0.0+recogdrive-stub"


def _unavailable(name):
    """Fail loudly if the stub is actually used."""
    raise RuntimeError(f"decord.{name} was called, but decord is only a stub here (env/stubs/decord).")


class VideoReader:
    """Stub that raises on construction."""

    def __init__(self, *args, **kwargs):
        _unavailable("VideoReader")


def cpu(*args, **kwargs):
    """Stub that raises."""
    _unavailable("cpu")

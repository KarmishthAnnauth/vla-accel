"""Placeholder for `decord` on the Orin.

navsim/agents/recogdrive/utils/internvl_preprocess.py does
`from decord import VideoReader, cpu` at module level, but ReCogDrive's
inference only ever calls that module's image functions (load_image,
dynamic_preprocess, build_transform); nothing in the planning path decodes a
video.  decord publishes no aarch64 wheel and building it needs the FFmpeg
development stack, so on the Jetson this stub satisfies the import instead.

start_recogdrive.sh puts it LAST on PYTHONPATH, so a real decord, if one is
ever installed in the image, wins.  Touching either name fails loudly rather
than pretending to work.
"""

__version__ = "0.0.0+recogdrive-ros-stub"


def _unavailable(name):
    raise RuntimeError(
        f"decord.{name} was called, but decord is only a stub on this machine "
        f"(recogdrive_env/stubs/decord). Video decoding is not part of ReCogDrive's "
        f"image inference; install a real decord build if you need it.")


class VideoReader:
    def __init__(self, *args, **kwargs):
        _unavailable("VideoReader")


def cpu(*args, **kwargs):
    _unavailable("cpu")

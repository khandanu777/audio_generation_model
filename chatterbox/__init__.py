try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("chatterbox-tts")
except Exception:
    __version__ = "0.0.0-local"


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
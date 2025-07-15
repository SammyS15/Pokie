from .pokie import get_device, pokie, pokie_bootstrap

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0-dev"


__all__ = (
    "get_device",
    "pokie",
    "pokie_bootstrap",
    "__version__",
)
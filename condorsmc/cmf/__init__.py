import logging
from importlib.metadata import PackageNotFoundError, version

logging.getLogger("condorsmc").addHandler(logging.NullHandler())

try:
    __version__ = version("condorsmc")
except PackageNotFoundError:
    __version__ = "UNKNOWN"
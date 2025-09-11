import logging
from importlib.metadata import PackageNotFoundError, version

logging.getLogger(__name__.split('.')[0]).addHandler(logging.NullHandler())

try:
    __version__ = version("condorsmc")
except PackageNotFoundError:
    __version__ = "UNKNOWN"

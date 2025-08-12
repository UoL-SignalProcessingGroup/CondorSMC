import logging

logging.getLogger("condorsmc").addHandler(logging.NullHandler())

try:
    from importlib import metadata as importlib_metadata

    __version__ = importlib_metadata.version("condorsmc")  # type: ignore
except ImportError:
    try:
        import pkg_resources

        __version__ = pkg_resources.get_distribution("condorsmc").version  # type: ignore
    except pkg_resources.DistributionNotFound:
        __version__ = "UNKNOWN"  # type: ignore
        pass
    pass
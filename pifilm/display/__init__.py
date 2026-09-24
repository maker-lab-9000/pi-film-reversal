"""Optional SPI LCD viewfinder for the Pi.

Hardware libraries are imported only inside the ``open_*`` factories so that
the package imports cleanly on a Mac and in the test suite.
"""


class DisplayError(Exception):
    """The display or touch panel could not be opened or driven."""

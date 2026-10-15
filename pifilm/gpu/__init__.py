"""The grade on the Pi's GPU: one shader pass instead of three CPU passes.

``shader`` holds the source, ``reference`` the same maths in NumPy (what the
tests and the Pi's self-test compare against). Nothing in this package loads a
graphics library at import time, so it imports on a Mac.
"""

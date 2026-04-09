"""FrameX compute backends.

``c_backend`` — C kernels compiled at first import via ctypes.
``C_AVAILABLE`` — True when compilation succeeded.
"""

from framex.backends.c_backend import C_AVAILABLE

__all__ = ["C_AVAILABLE"]

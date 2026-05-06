"""Back-compat shim. New code should import from `msa.routing`."""
from .routing import Route, route

__all__ = ["Route", "route"]

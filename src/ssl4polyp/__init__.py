"""Top-level package for SSL4POLYP."""

from ssl4polyp._compat import ensure_torch_container_abcs

# Ensure compatibility patches are applied as soon as the package is imported.
ensure_torch_container_abcs()

__all__ = [
    "classification",
    "models",
    "polypdb",
    "utils",
]

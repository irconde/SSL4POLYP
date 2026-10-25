"""Import the real PyYAML before any legacy test module installs its ``yaml`` stub.

Some legacy modules replace ``yaml`` in ``sys.modules`` only if it is not imported yet; in their original
location other test modules had always imported it first. Importing it here keeps that behaviour.
"""
import yaml  # noqa: F401

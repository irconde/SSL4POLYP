"""Test collection settings.

``tests/legacy/`` holds the tests of the v2 code base that already failed before campaign v3 (Python 3.9+
syntax, or expectations of the v2 reports and metric blocks). They are kept for reference and skipped by
default; run them with ``SSL4POLYP_LEGACY_TESTS=1 pytest tests/legacy``.
"""
import os

collect_ignore = [] if os.environ.get("SSL4POLYP_LEGACY_TESTS") == "1" else ["legacy"]

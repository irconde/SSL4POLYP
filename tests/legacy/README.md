# Legacy tests (v2)

Tests of the v2 code base that were already failing before campaign v3 (21 failures and 6 collection
errors at commit `d678b28`, before any v3 change; same set at the article state). They are excluded from
the default run (`tests/conftest.py`) and kept for reference:

```bash
SSL4POLYP_LEGACY_TESTS=1 pytest tests/legacy
```

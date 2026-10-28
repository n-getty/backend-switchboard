# Tests

`selftest.py` is a standalone script (no pytest, no network, no credentials). It starts mock upstreams and the router
in-process and checks routing, header and credential handling, the model rewrite, streaming latency, tool-search
flattening, and Argo's HTTP-200 quota notice.

```bash
python3 tests/selftest.py
```

CI runs it on Python 3.8 and 3.11. Real backends are not exercised in CI.

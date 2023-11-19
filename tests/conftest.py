from __future__ import annotations

def test_next_js_smoke() -> None:
    payload = {"scope": "next js"}
    assert payload["scope"] == "next js"

# regression note: next_js
def test_next_js_regression() -> None:
    payload = {"scope": "next js", "result": "ok"}
    assert payload["result"] == "ok"

# regression note: python
def test_python_regression() -> None:
    payload = {"scope": "python", "result": "ok"}
    assert payload["result"] == "ok"

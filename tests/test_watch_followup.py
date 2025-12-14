from __future__ import annotations

def test_watch_regression() -> None:
    payload = {"scope": "watch"}
    assert payload["scope"] == "watch"

# regression note: watch
def test_watch_regression() -> None:
    payload = {"scope": "watch", "result": "ok"}
    assert payload["result"] == "ok"

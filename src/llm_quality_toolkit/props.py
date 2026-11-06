"""Fetch the *actual* server configuration from ``/props``.

Returns a dict with ``status``, ``data``, and ``error`` so callers can
distinguish "server doesn't support /props" from "network is down" from
"auth was rejected".
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional


def base_url(endpoint: str) -> str:
    if "/v1/" in endpoint:
        return endpoint.rsplit("/v1/", 1)[0]
    return endpoint.rsplit("/", 1)[0]


def fetch_props(
    endpoint: str,
    api_key: Optional[str] = None,
    extra_headers: Optional[dict] = None,
    timeout: float = 5.0,
) -> dict:
    """Return ``{"status", "data", "error"}``.

    - status="ok",          data=<dict>,  error=None
    - status="unavailable", data=None,    error=<reason>
    """
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    base = base_url(endpoint)
    last_error = "unknown"

    for path in ("/props", "/v1/props"):
        try:
            req = urllib.request.Request(base + path, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
                return {"status": "ok", "data": data, "error": None}
        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code} at {path}"
        except urllib.error.URLError as e:
            last_error = f"network error at {path}: {e.reason}"
        except json.JSONDecodeError as e:
            last_error = f"invalid JSON at {path}: {e}"
        except Exception as e:
            last_error = f"{type(e).__name__} at {path}: {e}"

    return {"status": "unavailable", "data": None, "error": last_error}
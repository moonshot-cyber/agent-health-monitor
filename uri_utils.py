"""Shared URI resolution and name extraction utilities.

Used by erc8004_scan, celo_scan, and arc_scan to resolve agentURI / tokenURI
documents and extract agent names from the registration JSON.
"""

import base64
import json
import logging
import time

import requests

logger = logging.getLogger("ahm.uri_utils")

# IPFS gateway
IPFS_GATEWAY = "https://ipfs.io/ipfs/"

# HTTP fetch defaults
URI_FETCH_TIMEOUT = 10  # seconds


def resolve_uri(uri: str) -> tuple[dict | None, str]:
    """Fetch a single URI and return (json_data, error_string).

    Supports data: URIs (base64-encoded JSON), ipfs:// URIs (via public
    gateway), and http(s):// URIs.  Returns (None, error_tag) on any
    failure so callers can log error categories without crashing.
    """
    if not uri or not uri.strip():
        return None, "empty"

    url = uri.strip()

    # data: URI (base64 JSON, common in NFTs)
    if url.startswith("data:"):
        try:
            _, encoded = url.split(",", 1)
            decoded = base64.b64decode(encoded).decode("utf-8")
            return json.loads(decoded), ""
        except Exception as e:
            return None, f"data_uri_error: {str(e)[:60]}"

    # IPFS
    if url.startswith("ipfs://"):
        cid = url[7:]
        url = f"{IPFS_GATEWAY}{cid}"

    # Must be HTTP(S) at this point
    if not url.startswith("http"):
        return None, f"unsupported_scheme: {url[:40]}"

    try:
        resp = requests.get(
            url,
            timeout=URI_FETCH_TIMEOUT,
            headers={"Accept": "application/json", "User-Agent": "AHM-Scanner/1.0"},
        )
        resp.raise_for_status()
        return resp.json(), ""
    except requests.Timeout:
        return None, "timeout"
    except requests.HTTPError as e:
        return None, f"http_{e.response.status_code}"
    except requests.ConnectionError:
        return None, "connection_error"
    except json.JSONDecodeError:
        return None, "invalid_json"
    except Exception as e:
        return None, str(e)[:60]


def extract_name_from_json(data: dict) -> str:
    """Pull agent name from registration JSON.

    Checks keys: name, title, agent_name, agentName — returns the first
    non-empty string value found, truncated to 80 characters.
    """
    for key in ("name", "title", "agent_name", "agentName"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()[:80]
    return ""


def resolve_agent_names(
    wallets: list[dict],
    *,
    uri_key: str = "agent_uri",
    delay: float = 0.5,
    log: logging.Logger | None = None,
) -> dict[str, int]:
    """Resolve agent names from URIs stored in wallet metadata dicts.

    For each wallet, reads ``metadata[uri_key]``, resolves the URI to JSON
    via :func:`resolve_uri`, extracts the agent name via
    :func:`extract_name_from_json`, and stores it as ``metadata["agent_name"]``.

    Identical URIs are resolved only once (cached) to avoid redundant
    HTTP requests when multiple wallets share an agent registration.

    Returns a dict of ``error_type -> count`` for observability, matching
    the error-tracking pattern used by ``erc8004_scan.resolve_uris()``.
    """
    _log = log or logger
    errors_by_type: dict[str, int] = {}
    resolved_count = 0
    uri_cache: dict[str, tuple[dict | None, str]] = {}

    for w in wallets:
        meta = w.get("metadata") or {}
        uri = meta.get(uri_key, "")

        if not uri:
            errors_by_type["no_uri"] = errors_by_type.get("no_uri", 0) + 1
            continue

        # Cache to avoid resolving the same URI twice
        if uri not in uri_cache:
            data, err = resolve_uri(uri)
            uri_cache[uri] = (data, err)
            if delay > 0:
                time.sleep(delay)
        else:
            data, err = uri_cache[uri]

        if err:
            err_key = err.split(":")[0]
            errors_by_type[err_key] = errors_by_type.get(err_key, 0) + 1
        elif data:
            name = extract_name_from_json(data)
            if name:
                w.setdefault("metadata", {})["agent_name"] = name
                resolved_count += 1

    _log.info("URI resolution: %d/%d names extracted", resolved_count, len(wallets))
    if errors_by_type:
        _log.info(
            "URI errors: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(errors_by_type.items(), key=lambda x: -x[1])),
        )

    return errors_by_type

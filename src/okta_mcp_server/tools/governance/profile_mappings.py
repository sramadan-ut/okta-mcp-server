# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2026-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Read-only tools for finding where Okta user profile attributes are used.

Everything goes through ``okta_raw_request`` and works on plain JSON, not the
SDK's typed models. The typed models drop SAML attribute statement contents
and fail on some app types (e.g. Zapier), which would silently hide results.

Paging (management API, ``/api/v1``):

- Next pages come from the ``Link: <...>; rel="next"`` response header when
  the raw response exposes headers.
- Fallback when headers aren't available: if a page is full (``len == limit``),
  request the next page with ``after=<last item id>``.
- Next-page URLs are reduced to path + query and must stay under ``/api/v1/``,
  so the bearer token only ever goes to the configured org.

Resumability: there is one ``GET /api/v1/mappings/{id}`` per Okta<->app
mapping, which can outlast the MCP client's ~100s timeout in large orgs, and
some MCP proxies start a fresh server process per tool call. So the tool is
stateless: each call has a time budget and returns an opaque ``next_cursor``
that records the phase (``mappings`` then ``saml``) and the last processed
id. Mapping candidates are sorted by id so the order is the same in every
process. Mapping details are also cached in memory and in a per-org JSON file
under /tmp for 30 minutes, purely as a speedup: results never depend on the
cache. An HTTP 429 stops the call instead of retrying in a loop.
"""

import asyncio
import base64
import binascii
import bisect
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

from loguru import logger
from mcp.server.fastmcp import Context

from okta_mcp_server.server import mcp
from okta_mcp_server.utils.client import get_okta_client
from okta_mcp_server.utils.raw_request import okta_raw_request
from okta_mcp_server.utils.scope_guard import MissingScopeError, require_scopes
from okta_mcp_server.utils.serialization import json_response

_MGMT_BASE = "/api/v1"
_PAGE_LIMIT = 200
_MAX_PAGES = 100
_PAGE_DELAY_SECONDS = 0.1
#: Concurrent GET /api/v1/mappings/{id} calls. Kept low for rate limits.
_DETAIL_CONCURRENCY = 6

#: Time budget per call. MCP clients give up after ~100s, so the cap leaves headroom.
_DEFAULT_MAX_SECONDS = 60.0
_MAX_SECONDS_CAP = 80.0

#: How long cached mapping details are reused (memory and disk).
_CACHE_TTL_SECONDS = 30 * 60
#: Directory for the per-org mapping detail cache file.
_DISK_CACHE_DIR = Path("/tmp")

_CURSOR_VERSION = 1
_PHASE_MAPPINGS = "mappings"
_PHASE_SAML = "saml"
_PHASE_DONE = "done"

_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?')

# ---------------------------------------------------------------------------
# Mapping detail cache: a speedup only, never needed for correctness
# ---------------------------------------------------------------------------

#: mapping id -> (fetched_at wall-clock seconds, GET /api/v1/mappings/{id} body)
_DETAIL_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _clear_caches() -> None:
    _DETAIL_CACHE.clear()


def _is_fresh(fetched_at: float, now: float) -> bool:
    return 0 <= now - fetched_at < _CACHE_TTL_SECONDS


def _cached_detail(mapping_id: Any, now: float) -> Optional[Dict[str, Any]]:
    entry = _DETAIL_CACHE.get(mapping_id)
    if entry and _is_fresh(entry[0], now):
        return entry[1]
    return None


def _disk_cache_path(org_url: str) -> Path:
    digest = hashlib.sha256(org_url.rstrip("/").lower().encode()).hexdigest()[:16]
    return _DISK_CACHE_DIR / f"okta-mcp-profile-mapping-details-{digest}.json"


def _load_disk_cache(path: Path, now: float) -> int:
    """Merge fresh entries from ``path`` into the memory cache. Returns how many were added.

    Only reads a regular file owned by the current user (no symlinks), so
    another local user can't plant mapping details. Any error is ignored.
    """
    added = 0
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, encoding="utf-8") as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode) or (hasattr(os, "getuid") and st.st_uid != os.getuid()):
                return 0
            data = json.load(f)
        for mapping_id, (fetched_at, detail) in (data.get("entries") or {}).items():
            fetched_at = float(fetched_at)
            current = _DETAIL_CACHE.get(mapping_id)
            if isinstance(detail, dict) and _is_fresh(fetched_at, now) and (not current or current[0] < fetched_at):
                _DETAIL_CACHE[mapping_id] = (fetched_at, detail)
                added += 1
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug(f"Ignoring unreadable profile mapping cache {path}: {type(e).__name__}: {e}")
    return added


def _save_disk_cache(path: Path, now: float) -> None:
    """Atomically write fresh memory-cache entries (merged with the file) to ``path``. Errors are ignored."""
    tmp_name = None
    try:
        _load_disk_cache(path, now)  # keep entries other processes wrote meanwhile
        entries = {k: [ts, d] for k, (ts, d) in _DETAIL_CACHE.items() if _is_fresh(ts, now)}
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")  # mode 0600
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "entries": entries}, f, separators=(",", ":"))
        os.replace(tmp_name, path)
        tmp_name = None
    except Exception as e:
        logger.debug(f"Could not write profile mapping cache {path}: {type(e).__name__}: {e}")
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------


class _CursorError(ValueError):
    pass


def _args_fingerprint(attributes: List[str], include_saml: bool, app_name: Optional[str]) -> str:
    raw = json.dumps([sorted(attributes), bool(include_saml), (app_name or "").strip().lower()])
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _new_cursor_state(fingerprint: str) -> Dict[str, Any]:
    return {
        "v": _CURSOR_VERSION,
        "args": fingerprint,
        "phase": _PHASE_MAPPINGS,
        "after": None,
        "checked": 0,
        "failed": 0,
    }


def _encode_cursor(state: Dict[str, Any]) -> str:
    raw = json.dumps(state, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str, fingerprint: str) -> Dict[str, Any]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        state = json.loads(raw)
    except (binascii.Error, ValueError, UnicodeDecodeError) as e:
        raise _CursorError("Invalid cursor: pass next_cursor exactly as returned by the previous call.") from e
    if (
        not isinstance(state, dict)
        or state.get("v") != _CURSOR_VERSION
        or state.get("phase") not in (_PHASE_MAPPINGS, _PHASE_SAML)
        or not isinstance(state.get("after"), (str, type(None)))
        or not isinstance(state.get("checked"), int)
        or not isinstance(state.get("failed"), int)
    ):
        raise _CursorError("Invalid cursor: pass next_cursor exactly as returned by the previous call.")
    if state.get("args") != fingerprint:
        raise _CursorError(
            "This cursor belongs to a scan with different arguments. Pass the same attributes, "
            "include_saml and app_name as the call that returned it, or omit cursor to start over."
        )
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_result(err: Any) -> Dict[str, Any]:
    if not isinstance(err, MissingScopeError):
        return {"error": str(err)}
    scopes = ", ".join(sorted(err.scopes))
    return {
        "error": (
            f"Okta denied the request (HTTP {err.status}"
            + (f" {err.error_code}" if err.error_code else "")
            + (f": {err.error_summary}" if err.error_summary else "")
            + f"). Check that {scopes} is in OKTA_SCOPES and granted to the service app, "
            "and that the app has an admin role that can read profile mappings and applications."
        ),
        "status": err.status,
        "errorCode": err.error_code,
        "errorSummary": err.error_summary,
        "required_scopes": sorted(err.scopes),
    }


def _is_rate_limited(err: Any) -> bool:
    """True for an Okta HTTP 429, after the SDK's own retries are used up."""
    if getattr(err, "status", None) == 429 or getattr(err, "error_code", None) == "E0000047":
        return True
    # The SDK returns a plain Exception when a 429 lacks X-Rate-Limit-Reset / Date.
    return str(err).startswith("429 ")


def _mgmt_url(path: str, params: Optional[Dict[str, Any]] = None) -> str:
    query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
    return f"{_MGMT_BASE}{path}" + (f"?{query}" if query else "")


def _safe_next_url(href: Optional[str]) -> Optional[str]:
    """Reduce an absolute next link to path + query under /api/v1."""
    if not href:
        return None
    parsed = urlparse(href)
    if not parsed.path.startswith(f"{_MGMT_BASE}/"):
        logger.warning(f"Ignoring next link outside {_MGMT_BASE}: {parsed.path}")
        return None
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


def _response_headers(response: Any) -> Any:
    """Headers of the raw response, or None if the response doesn't expose them.

    ``okta_raw_request`` returns the SDK's ``aiohttp.ClientResponse``, whose
    ``headers`` is a case-insensitive ``CIMultiDictProxy``.
    """
    headers = getattr(response, "headers", None)
    if headers is None and isinstance(response, dict):
        headers = response.get("headers")
    return headers


def _link_header_values(headers: Any) -> List[str]:
    if not headers:
        return []
    if hasattr(headers, "getall"):  # aiohttp CIMultiDict: one entry per Link header
        try:
            return list(headers.getall("Link", []))
        except Exception:
            pass
    for key in ("Link", "link"):
        try:
            value = headers.get(key)
        except Exception:
            value = None
        if value:
            return value if isinstance(value, list) else [value]
    return []


def _next_from_link_header(response: Any) -> Tuple[bool, Optional[str]]:
    """Return (headers_available, next_url).

    When headers are available, a missing ``rel="next"`` link means this is the
    last page, so callers must not fall back to ``after=`` paging.
    """
    headers = _response_headers(response)
    if headers is None:
        return False, None
    values = _link_header_values(headers)
    for value in values:
        for match in _LINK_NEXT_RE.finditer(value):
            return True, _safe_next_url(match.group(1))
    return True, None


def _has_next_page(response: Any, page: List[Any]) -> bool:
    has_headers, next_url = _next_from_link_header(response)
    if has_headers:
        return next_url is not None
    return len(page) >= _PAGE_LIMIT


def _with_after(url: str, after: str) -> str:
    parsed = urlparse(url)
    params = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
    params["after"] = after
    return parsed.path + "?" + urlencode(params)


async def _get_all(
    ctx: Context, client, url: str, what: str, deadline: Optional[float] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch every page of a /api/v1 list. Returns (items, info).

    ``deadline`` (``time.monotonic()`` value) stops paging before a new page
    once it has passed; the first page is always fetched.
    """
    items: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {"stopped_early": False, "stop_reason": None, "pages_fetched": 0}
    seen = set()
    next_url: Optional[str] = url

    while next_url:
        if next_url in seen:
            info.update(stopped_early=True, stop_reason="Pagination cursor did not advance")
            break
        if info["pages_fetched"] >= _MAX_PAGES:
            info.update(stopped_early=True, stop_reason=f"Reached maximum page limit ({_MAX_PAGES})")
            break
        if info["pages_fetched"] and deadline is not None and time.monotonic() >= deadline:
            info.update(stopped_early=True, stop_reason="Time budget spent")
            break
        seen.add(next_url)
        if info["pages_fetched"]:
            await asyncio.sleep(_PAGE_DELAY_SECONDS)

        response, body, err = await okta_raw_request(client, "GET", next_url)
        if err:
            if info["pages_fetched"] == 0:
                raise _ApiError(err)
            info.update(stopped_early=True, stop_reason=f"API error: {_error_result(err)['error']}")
            break

        page = body if isinstance(body, list) else []
        items.extend(p for p in page if isinstance(p, dict))
        info["pages_fetched"] += 1
        try:
            await ctx.info(f"Fetching {what}... {len(items)} so far")
        except Exception:
            pass

        has_headers, header_next = _next_from_link_header(response)
        if has_headers:
            next_url = header_next
        elif len(page) >= _PAGE_LIMIT and isinstance(page[-1], dict) and page[-1].get("id"):
            next_url = _with_after(next_url, page[-1]["id"])
        else:
            next_url = None

    info["total_items"] = len(items)
    return items, info


class _ApiError(Exception):
    def __init__(self, err: Any):
        super().__init__(str(err))
        self.err = err


def _attribute_pattern(attributes: Iterable[str]) -> re.Pattern:
    names = "|".join(re.escape(a) for a in attributes)
    # Case-sensitive, like Okta Expression Language. Matches user.Team but not user.TeamLead.
    return re.compile(rf"\buser\.({names})(?![A-Za-z0-9_])")


def _mapping_matches(
    summary: Dict[str, Any], detail: Dict[str, Any], pattern: re.Pattern, wanted: set
) -> List[Dict[str, Any]]:
    src, tgt = summary.get("source") or {}, summary.get("target") or {}
    okta_to_app = src.get("type") == "user" and tgt.get("type") == "appuser"
    app_to_okta = src.get("type") == "appuser" and tgt.get("type") == "user"
    rows: List[Dict[str, Any]] = []
    for attr, prop in (detail.get("properties") or {}).items():
        prop = prop or {}
        expr = prop.get("expression") or ""
        if okta_to_app and pattern.search(expr):
            rows.append(
                {
                    "app": tgt.get("name"),
                    "app_instance_id": tgt.get("id"),
                    "direction": "okta_to_app",
                    "via": "profile_mapping",
                    "app_attribute": attr,
                    "okta_attributes": sorted(set(pattern.findall(expr))),
                    "expression": expr,
                    "push_status": prop.get("pushStatus"),
                    "mapping_id": summary.get("id"),
                }
            )
        elif app_to_okta and attr in wanted:
            rows.append(
                {
                    "app": src.get("name"),
                    "app_instance_id": src.get("id"),
                    "direction": "app_to_okta",
                    "via": "profile_mapping",
                    "okta_attributes": [attr],
                    "expression": expr,
                    "push_status": prop.get("pushStatus"),
                    "mapping_id": summary.get("id"),
                }
            )
    return rows


def _saml_matches(app: Dict[str, Any], pattern: re.Pattern) -> List[Dict[str, Any]]:
    sign_on = (app.get("settings") or {}).get("signOn") or {}
    rows: List[Dict[str, Any]] = []
    for stmt in sign_on.get("attributeStatements") or []:
        if not isinstance(stmt, dict):
            continue
        for value in stmt.get("values") or []:
            if isinstance(value, str) and pattern.search(value):
                rows.append(
                    {
                        "app": app.get("label"),
                        "app_instance_id": app.get("id"),
                        "app_status": app.get("status"),
                        "direction": "okta_to_app",
                        "via": "saml_attribute_statement",
                        "app_attribute": stmt.get("name"),
                        "okta_attributes": sorted(set(pattern.findall(value))),
                        "expression": value,
                    }
                )
    return rows


def _mapping_app_name(summary: Dict[str, Any]) -> str:
    """Name of the app side of a user<->appuser mapping."""
    for side in ("source", "target"):
        end = summary.get(side) or {}
        if end.get("type") == "appuser":
            return str(end.get("name") or "")
    return ""


def _app_name_matches(needle: Optional[str], *names: Any) -> bool:
    return not needle or any(needle in str(n or "").lower() for n in names)


# ---------------------------------------------------------------------------
# Phase 1: profile mappings
# ---------------------------------------------------------------------------


async def _fetch_mapping_details(
    client, pending: List[Dict[str, Any]], deadline: float
) -> Tuple[Dict[int, Tuple[str, Any]], Dict[str, Any]]:
    """Resolve ``pending`` mappings in order until done, the deadline passes, or Okta returns 429.

    Workers take items strictly in list order. Cached items resolve without a
    request (even past the deadline); uncached ones are only started before
    the deadline, and a request still in flight at the deadline is cancelled.
    Returns ({index: (kind, payload)}, stats) with kind one of ``ok`` (payload
    = detail), ``forbidden``, ``gone`` or ``failed`` (payload = error).
    """
    resolved: Dict[int, Tuple[str, Any]] = {}
    stats: Dict[str, Any] = {"next": 0, "timed_out": False, "rate_limited": False, "fetched": 0, "from_cache": 0}

    async def worker() -> None:
        while not stats["rate_limited"]:
            i = stats["next"]
            if i >= len(pending):
                return
            mapping_id = pending[i].get("id")
            cached = _cached_detail(mapping_id, time.time())
            if cached is not None:
                stats["next"] += 1
                stats["from_cache"] += 1
                resolved[i] = ("ok", cached)
                continue
            left = deadline - time.monotonic()
            if left <= 0:
                stats["timed_out"] = True
                return
            stats["next"] += 1
            try:
                _, detail, err = await asyncio.wait_for(
                    okta_raw_request(client, "GET", _mgmt_url(f"/mappings/{mapping_id}")), timeout=left
                )
            except asyncio.TimeoutError:
                stats["timed_out"] = True
                return
            stats["fetched"] += 1
            if err:
                if _is_rate_limited(err):
                    stats["rate_limited"] = True
                    return
                status = getattr(err, "status", None)
                resolved[i] = (
                    ("forbidden", err) if status == 403 else ("gone", err) if status == 404 else ("failed", err)
                )
                continue
            if not isinstance(detail, dict):
                resolved[i] = ("failed", "empty response body")
                continue
            _DETAIL_CACHE[mapping_id] = (time.time(), detail)
            resolved[i] = ("ok", detail)

    await asyncio.gather(*(worker() for _ in range(_DETAIL_CONCURRENCY)))
    return resolved, stats


async def _mappings_phase(
    ctx: Context,
    client,
    state: Dict[str, Any],
    needle: Optional[str],
    pattern: re.Pattern,
    wanted: set,
    deadline: float,
    org_url: Optional[str],
    out: Dict[str, Any],
) -> bool:
    """Process mapping candidates after ``state["after"]``. Updates ``state`` and ``out``; returns True when done."""
    mappings, info = await _get_all(
        ctx, client, _mgmt_url("/mappings", {"limit": _PAGE_LIMIT}), "profile mappings", deadline
    )
    if info["stopped_early"]:
        # Without the full list the id order isn't known, so don't advance.
        out["errors"].append(f"Profile mappings list incomplete ({info['stop_reason']}); no mappings checked.")
        out["retry_same_cursor"] = True
        return False

    candidates = sorted(
        (
            m
            for m in mappings
            if m.get("id")
            and {(m.get("source") or {}).get("type"), (m.get("target") or {}).get("type")} == {"user", "appuser"}
            and _app_name_matches(needle, _mapping_app_name(m))
        ),
        key=lambda m: str(m["id"]),
    )
    ids = [str(m["id"]) for m in candidates]
    start = bisect.bisect_right(ids, state["after"]) if state["after"] is not None else 0
    pending = candidates[start:]
    out["progress_total"] = len(candidates)

    disk_path = _disk_cache_path(org_url) if org_url else None
    if disk_path and pending:
        _load_disk_cache(disk_path, time.time())

    resolved, stats = await _fetch_mapping_details(client, pending, deadline)
    out["mappings_fetched"] = stats["fetched"]
    out["mappings_from_cache"] = stats["from_cache"]
    out["rate_limited"] = stats["rate_limited"]
    if disk_path and stats["fetched"]:
        _save_disk_cache(disk_path, time.time())

    # Only the contiguous resolved prefix counts, so the cursor never skips a mapping.
    processed = 0
    for i, summary in enumerate(pending):
        if i not in resolved:
            break
        kind, payload = resolved[i]
        mapping_id, app = str(summary["id"]), _mapping_app_name(summary)
        if kind == "ok":
            out["matches"].extend(_mapping_matches(summary, payload, pattern, wanted))
        elif kind == "forbidden":
            out["forbidden_mappings"].append({"mapping_id": mapping_id, "app": app})
        elif kind == "failed":
            reason = _error_result(payload)["error"] if not isinstance(payload, str) else payload
            out["errors"].append(f"mapping {mapping_id} ({app}): {reason}")
            state["failed"] += 1
        state["after"] = mapping_id
        processed += 1

    state["checked"] = start + processed
    return start + processed == len(candidates)


# ---------------------------------------------------------------------------
# Phase 2: SAML attribute statements
# ---------------------------------------------------------------------------


async def _saml_phase(
    ctx: Context,
    client,
    state: Dict[str, Any],
    needle: Optional[str],
    pattern: re.Pattern,
    deadline: float,
    out: Dict[str, Any],
) -> bool:
    """Page /api/v1/apps with after=<last app id>, one whole page at a time. Returns True when done."""
    first = True
    while True:
        if not first:
            await asyncio.sleep(_PAGE_DELAY_SECONDS)
        first = False
        left = deadline - time.monotonic()
        if left <= 0:
            return False
        url = _mgmt_url("/apps", {"limit": _PAGE_LIMIT, "after": state["after"]})
        try:
            response, body, err = await asyncio.wait_for(okta_raw_request(client, "GET", url), timeout=left)
        except asyncio.TimeoutError:
            return False
        if err:
            if _is_rate_limited(err):
                out["rate_limited"] = True
            else:
                out["errors"].append(f"applications page after {state['after']!r}: {_error_result(err)['error']}")
                out["retry_same_cursor"] = True
            return False

        page = [a for a in (body if isinstance(body, list) else []) if isinstance(a, dict)]
        for app in page:
            if _app_name_matches(needle, app.get("label"), app.get("name")):
                out["matches"].extend(_saml_matches(app, pattern))
        state["checked"] += len(page)
        if page and page[-1].get("id"):
            state["after"] = str(page[-1]["id"])
        try:
            await ctx.info(f"Scanning applications for SAML attribute statements... {state['checked']} so far")
        except Exception:
            pass
        if not page or not page[-1].get("id") or not _has_next_page(response, page):
            return True


# ---------------------------------------------------------------------------
# One chunk of the scan
# ---------------------------------------------------------------------------


async def _scan(
    ctx: Context,
    client,
    attributes: List[str],
    include_saml: bool = True,
    app_name: Optional[str] = None,
    max_seconds: float = _DEFAULT_MAX_SECONDS,
    cursor: Optional[str] = None,
    org_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one time-boxed chunk of the scan, starting at ``cursor``. Raises _CursorError for a bad cursor."""
    started = time.monotonic()
    deadline = started + max_seconds
    fingerprint = _args_fingerprint(attributes, include_saml, app_name)
    state = _decode_cursor(cursor, fingerprint) if cursor else _new_cursor_state(fingerprint)
    needle = (app_name or "").strip().lower() or None
    pattern = _attribute_pattern(attributes)
    out: Dict[str, Any] = {
        "matches": [],
        "errors": [],
        "forbidden_mappings": [],
        "rate_limited": False,
        "retry_same_cursor": False,
        "progress_total": None,
        "mappings_fetched": 0,
        "mappings_from_cache": 0,
    }

    if state["phase"] == _PHASE_MAPPINGS:
        done = await _mappings_phase(ctx, client, state, needle, pattern, set(attributes), deadline, org_url, out)
        if done:
            state.update(phase=_PHASE_SAML if include_saml else _PHASE_DONE, after=None, checked=0)
            out["progress_total"] = None
    if state["phase"] == _PHASE_SAML and not out["rate_limited"] and time.monotonic() < deadline:
        if await _saml_phase(ctx, client, state, needle, pattern, deadline, out):
            state["phase"] = _PHASE_DONE

    finished = state["phase"] == _PHASE_DONE
    matches = sorted(
        out["matches"], key=lambda r: (str(r.get("app") or "").lower(), r["via"], str(r.get("app_attribute") or ""))
    )
    result: Dict[str, Any] = {
        "attributes": attributes,
        "app_name": app_name,
        "matches": matches,
        "apps_by_attribute": {
            a: sorted({str(r["app"]) for r in matches if a in r["okta_attributes"]}) for a in attributes
        },
        "total_matches": len(matches),
        "forbidden_mappings": out["forbidden_mappings"],
        "errors": out["errors"],
        "progress": {
            "phase": state["phase"],
            "checked_so_far": state["checked"],
            "total": out["progress_total"] if state["phase"] == _PHASE_MAPPINGS else None,
            "failed_so_far": state["failed"],
        },
        "next_cursor": None if finished else _encode_cursor(state),
        "complete": finished and state["failed"] == 0,
        "mappings_fetched": out["mappings_fetched"],
        "mappings_from_cache": out["mappings_from_cache"],
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    if out["rate_limited"]:
        result["errors"].append("Okta rate limit hit (HTTP 429); stopped early.")
    if not finished:
        wait = " Wait about a minute first because Okta rate-limited this call." if out["rate_limited"] else ""
        retry = " The failed step will be retried." if out["retry_same_cursor"] else ""
        result["message"] = (
            f"Partial result ({state['phase']} phase). Call find_profile_attribute_usage again with the same "
            "arguments and cursor=next_cursor, add this call's matches to the ones you already have, and "
            f"repeat until next_cursor is null.{wait}{retry}"
        )
    elif state["failed"]:
        result["message"] = (
            f"Scan finished, but {state['failed']} mapping(s) could not be read (see errors in earlier "
            "calls), so the result may be missing apps."
        )
    return result


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool()
@require_scopes("okta.profileMappings.read", "okta.apps.read")
@json_response
async def find_profile_attribute_usage(
    ctx: Context,
    attributes: List[str],
    include_saml: bool = True,
    app_name: Optional[str] = None,
    max_seconds: float = _DEFAULT_MAX_SECONDS,
    cursor: Optional[str] = None,
) -> dict:
    """Find which apps use given Okta user profile attributes.

    Use this for questions like "which apps use the Team field?", "what breaks
    if I rename or delete attribute X?", or "which app populates X?". Checks:

    1. Okta -> app profile mappings whose expressions read the attribute
       (provisioning / profile push). ``direction == "okta_to_app"``.
    2. App -> Okta profile mappings that write the attribute (e.g. an HRIS
       import). ``direction == "app_to_okta"``.
    3. SAML attribute statements whose expressions read the attribute
       (``via == "saml_attribute_statement"``), including inactive apps.

    Not covered: OIDC custom claims on authorization servers, group rules,
    Okta Workflows, and expressions that reference the attribute indirectly.

    PAGED — CALL AGAIN WITH next_cursor UNTIL IT IS NULL. Large orgs need
    several calls. Each call works for at most ``max_seconds`` and returns
    ONLY the matches found in that chunk, plus ``next_cursor``. While
    ``next_cursor`` is not null, call this tool again with the SAME
    attributes, include_saml and app_name and ``cursor=next_cursor``. Keep
    every call's ``matches`` and ``forbidden_mappings`` and combine them; do
    not answer until ``next_cursor`` is null, or say the result is partial.
    If ``errors`` mentions a rate limit (HTTP 429), wait about a minute
    before the next call. ``progress`` shows the phase (mappings, then saml,
    then done) and how far it got.

    ``forbidden_mappings`` lists mappings the service app may not read (HTTP
    403, usually an admin role limited to some apps). They don't stop the
    scan, but those apps weren't checked; mention them in the answer.

    Read-only. Pass ``app_name`` when the question is about specific apps so
    only those are read.

    Parameters:
        attributes (list[str], required): Okta user profile variable names,
            case-sensitive, e.g. ["Team", "timeType"]. Check exact names
            with get_user_profile_attributes if unsure.
        include_saml (bool, optional): Also scan SAML attribute statements.
            Default: True.
        app_name (str, optional): Only check apps whose name contains this
            text (case-insensitive): the app name on profile mappings, and the
            label or name on SAML apps. Default: all apps.
        max_seconds (float, optional): Time budget for this call. Default 60,
            capped at 80.
        cursor (str, optional): ``next_cursor`` from the previous call.
            Omit to start a new scan.

    Returns:
        Dict containing:
        - matches: This chunk's rows (app, direction, via, app_attribute,
          okta_attributes, expression, push_status)
        - apps_by_attribute: {attribute: [app names]} for this chunk only
        - forbidden_mappings: [{mapping_id, app}] this chunk could not read (403)
        - next_cursor: Pass back as ``cursor``; null when the scan is finished
        - progress: {phase, checked_so_far, total, failed_so_far}; total is
          null in the saml phase (the app count isn't known in advance)
        - complete: True only on the final chunk, when no mapping failed
        - message: What to do next, when the scan isn't finished or had failures
        - errors, total_matches, mappings_fetched, mappings_from_cache,
          elapsed_seconds
    """
    attributes = [a.strip() for a in (attributes or []) if a and a.strip()]
    if not attributes:
        return {"error": 'Provide at least one attribute name, e.g. ["Team", "timeType"].'}
    invalid = [a for a in attributes if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", a)]
    if invalid:
        return {"error": f"Invalid attribute names (use the variable name, not the display name): {invalid}"}
    try:
        budget = float(max_seconds)
    except (TypeError, ValueError):
        budget = _DEFAULT_MAX_SECONDS
    budget = min(max(budget, 1.0), _MAX_SECONDS_CAP)

    logger.info(
        f"Finding profile attribute usage for {attributes} (include_saml={include_saml}, "
        f"app_name={app_name!r}, max_seconds={budget}, resumed={bool(cursor)})"
    )
    manager = ctx.request_context.lifespan_context.okta_auth_manager

    try:
        client = await get_okta_client(manager)
        result = await _scan(
            ctx, client, attributes, include_saml, app_name, budget, cursor, getattr(manager, "org_url", None)
        )
        logger.info(
            f"Profile attribute usage chunk for {attributes}: {result['total_matches']} matches, "
            f"progress={result['progress']}, complete={result['complete']}"
        )
        return result
    except _CursorError as e:
        return {"error": str(e)}
    except _ApiError as e:
        logger.error(f"Okta API error while finding profile attribute usage: {e.err}")
        return _error_result(e.err)
    except Exception as e:
        logger.error(f"Exception while finding profile attribute usage: {type(e).__name__}: {e}")
        return {"error": str(e)}

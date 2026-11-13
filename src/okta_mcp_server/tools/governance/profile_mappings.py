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
mapping, which can outlast the MCP client's ~100s timeout in large orgs. Each
call has a time budget, mapping details and the apps list are cached in memory
for 30 minutes, and a partial result tells the caller to call again. An HTTP
429 stops the scan instead of retrying in a loop.
"""

import asyncio
import re
import time
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

#: How long cached mapping details and the apps list are reused.
_CACHE_TTL_SECONDS = 30 * 60

_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?')

# ---------------------------------------------------------------------------
# Caches (module-level, in-memory, per server process / configured org)
# ---------------------------------------------------------------------------

#: mapping id -> (fetched_at monotonic seconds, GET /api/v1/mappings/{id} body)
_DETAIL_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
#: Full, unfiltered GET /api/v1/apps list: {"fetched_at": float, "apps": [...]}
_APPS_CACHE: Dict[str, Any] = {}


def _clear_caches() -> None:
    _DETAIL_CACHE.clear()
    _APPS_CACHE.clear()


def _is_fresh(fetched_at: float, now: float) -> bool:
    return now - fetched_at < _CACHE_TTL_SECONDS


def _prune_detail_cache(now: float) -> None:
    for mapping_id in [k for k, (ts, _) in _DETAIL_CACHE.items() if not _is_fresh(ts, now)]:
        del _DETAIL_CACHE[mapping_id]


def _cached_detail(mapping_id: Any, now: float) -> Optional[Dict[str, Any]]:
    entry = _DETAIL_CACHE.get(mapping_id)
    if entry and _is_fresh(entry[0], now):
        return entry[1]
    return None


def _cached_apps(now: float) -> Optional[List[Dict[str, Any]]]:
    if _APPS_CACHE and _is_fresh(_APPS_CACHE["fetched_at"], now):
        return _APPS_CACHE["apps"]
    return None


def _is_rate_limited(err: Any) -> bool:
    """True for an Okta HTTP 429, after the SDK's own retries are used up."""
    if getattr(err, "status", None) == 429 or getattr(err, "error_code", None) == "E0000047":
        return True
    # The SDK returns a plain Exception when a 429 lacks X-Rate-Limit-Reset / Date.
    return str(err).startswith("429 ")


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


async def _fetch_details(
    client, to_fetch: List[Dict[str, Any]], deadline: float, errors: List[str]
) -> Tuple[bool, bool]:
    """Fetch and cache mapping details until done, the deadline passes, or Okta returns 429.

    Workers stop starting new requests once the deadline passes; a request
    still in flight at the deadline is cancelled and left for the next call.
    Returns (timed_out, rate_limited).
    """
    queue = iter(to_fetch)
    state = {"timed_out": False, "rate_limited": False}

    async def worker() -> None:
        while not state["rate_limited"]:
            left = deadline - time.monotonic()
            if left <= 0:
                state["timed_out"] = True
                return
            summary = next(queue, None)
            if summary is None:
                return
            mapping_id = summary.get("id")
            try:
                _, detail, err = await asyncio.wait_for(
                    okta_raw_request(client, "GET", _mgmt_url(f"/mappings/{mapping_id}")), timeout=left
                )
            except asyncio.TimeoutError:
                state["timed_out"] = True
                return
            if err:
                if _is_rate_limited(err):
                    if not state["rate_limited"]:
                        errors.append(
                            "Okta rate limit hit (HTTP 429) while reading mapping details; stopped early. "
                            "Wait about a minute, then call again with the same arguments to continue."
                        )
                    state["rate_limited"] = True
                    return
                errors.append(f"mapping {mapping_id}: {_error_result(err)['error']}")
                continue
            if not isinstance(detail, dict):
                errors.append(f"mapping {mapping_id}: empty body")
                continue
            _DETAIL_CACHE[mapping_id] = (time.monotonic(), detail)

    await asyncio.gather(*(worker() for _ in range(_DETAIL_CONCURRENCY)))
    return state["timed_out"], state["rate_limited"]


async def _scan(
    ctx: Context,
    client,
    attributes: List[str],
    include_saml: bool = True,
    app_name: Optional[str] = None,
    max_seconds: float = _DEFAULT_MAX_SECONDS,
) -> Dict[str, Any]:
    started = time.monotonic()
    deadline = started + max_seconds
    pattern = _attribute_pattern(attributes)
    wanted = set(attributes)
    needle = (app_name or "").strip().lower() or None
    matches: List[Dict[str, Any]] = []
    errors: List[str] = []
    pagination: Dict[str, Any] = {}
    _prune_detail_cache(started)

    # 1 + 2. Profile mappings in both directions between Okta users and apps.
    mappings, pagination["mappings"] = await _get_all(
        ctx, client, _mgmt_url("/mappings", {"limit": _PAGE_LIMIT}), "profile mappings", deadline
    )
    candidates = [
        m
        for m in mappings
        if {(m.get("source") or {}).get("type"), (m.get("target") or {}).get("type")} == {"user", "appuser"}
        and _app_name_matches(needle, _mapping_app_name(m))
    ]

    # 3. SAML attribute statements. The apps list is cached, so this hits Okta once per TTL.
    apps: List[Dict[str, Any]] = []
    saml_from_cache = False
    if include_saml:
        cached = _cached_apps(started)
        if cached is not None:
            apps, saml_from_cache = cached, True
            pagination["apps"] = {"stopped_early": False, "from_cache": True, "total_items": len(apps)}
        else:
            try:
                apps, pagination["apps"] = await _get_all(
                    ctx, client, _mgmt_url("/apps", {"limit": _PAGE_LIMIT}), "applications", deadline
                )
                if not pagination["apps"]["stopped_early"]:
                    _APPS_CACHE.update(fetched_at=time.monotonic(), apps=apps)
            except _ApiError as e:
                rate_limited = _is_rate_limited(e.err)
                reason = "Okta rate limit hit (HTTP 429)" if rate_limited else _error_result(e.err)["error"]
                errors.append(f"applications list: {reason}")
                pagination["apps"] = {"stopped_early": True, "stop_reason": reason, "pages_fetched": 0}
        for app in apps:
            if _app_name_matches(needle, app.get("label"), app.get("name")):
                matches.extend(_saml_matches(app, pattern))

    # Mapping details: reuse the cache, fetch the rest within the time budget.
    to_fetch = [m for m in candidates if _cached_detail(m.get("id"), started) is None]
    from_cache = len(candidates) - len(to_fetch)
    timed_out, rate_limited = False, False
    if to_fetch:
        try:
            await ctx.info(f"Reading {len(to_fetch)} mapping details ({from_cache} cached)...")
        except Exception:
            pass
        timed_out, rate_limited = await _fetch_details(client, to_fetch, deadline, errors)

    now = time.monotonic()
    checked = 0
    for summary in candidates:
        detail = _cached_detail(summary.get("id"), now)
        if detail is not None:
            checked += 1
            matches.extend(_mapping_matches(summary, detail, pattern, wanted))
    remaining = len(candidates) - checked
    listing_cut_short = any(p.get("stopped_early") for p in pagination.values())

    matches.sort(key=lambda r: (str(r.get("app") or "").lower(), r["via"], str(r.get("app_attribute") or "")))
    by_attribute = {a: sorted({str(r["app"]) for r in matches if a in r["okta_attributes"]}) for a in attributes}
    result: Dict[str, Any] = {
        "attributes": attributes,
        "app_name": app_name,
        "apps_by_attribute": by_attribute,
        "matches": matches,
        "total_matches": len(matches),
        "mappings_total": len(candidates),
        "mappings_checked": checked,
        "mappings_remaining": remaining,
        "mappings_from_cache": from_cache,
        "saml_apps_from_cache": saml_from_cache,
        "errors": errors,
        "pagination_info": pagination,
        "elapsed_seconds": round(now - started, 1),
        "complete": not errors and not remaining and not listing_cut_short,
    }
    if timed_out or rate_limited or (listing_cut_short and not errors):
        why = "Okta rate-limited the scan" if rate_limited else "the time budget ran out"
        result["message"] = (
            f"Partial result: {why} with {remaining} of {len(candidates)} mappings not yet checked. "
            "Progress is cached, so call find_profile_attribute_usage again with the same arguments "
            "to continue; repeat until complete is true."
            + (" Wait about a minute first because of the rate limit." if rate_limited else "")
        )
    elif not result["complete"]:
        result["message"] = (
            "Partial result: some requests failed (see errors). Calling again with the same arguments "
            "retries only the failed parts."
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

    RESUMABLE — CALL AGAIN UNTIL complete IS TRUE. Each call works for at most
    ``max_seconds``, then returns what it has. Mapping details and the apps
    list are cached for 30 minutes, so each repeat call continues where the
    last one stopped. If the result has ``complete: false`` and
    ``mappings_remaining > 0``, call this tool again with the SAME arguments
    (follow ``message``) and keep going until ``complete`` is true before
    answering. If you must answer earlier, say the result is partial. If
    ``errors`` mentions a rate limit (HTTP 429), wait about a minute first.

    Read-only. Makes one request per Okta<->app mapping (none for mappings
    already cached). Pass ``app_name`` when the question is about specific
    apps so only those are read.

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

    Returns:
        Dict containing:
        - apps_by_attribute: {attribute: [app names]}
        - matches: One row per app/attribute/expression with direction, via,
          app_attribute, expression and push_status
        - total_matches
        - mappings_total, mappings_checked, mappings_remaining,
          mappings_from_cache, saml_apps_from_cache
        - complete: False if any part of the scan failed, was cut short, or
          still has mappings remaining
        - message: Present when the result is partial; says what to do next
        - errors, pagination_info, elapsed_seconds
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
        f"Finding profile attribute usage for {attributes} "
        f"(include_saml={include_saml}, app_name={app_name!r}, max_seconds={budget})"
    )
    manager = ctx.request_context.lifespan_context.okta_auth_manager

    try:
        client = await get_okta_client(manager)
        result = await _scan(ctx, client, attributes, include_saml, app_name, budget)
        logger.info(
            f"Profile attribute usage for {attributes}: {result['total_matches']} matches, "
            f"{result['mappings_remaining']} mappings remaining, complete={result['complete']}"
        )
        return result
    except _ApiError as e:
        logger.error(f"Okta API error while finding profile attribute usage: {e.err}")
        return _error_result(e.err)
    except Exception as e:
        logger.error(f"Exception while finding profile attribute usage: {type(e).__name__}: {e}")
        return {"error": str(e)}

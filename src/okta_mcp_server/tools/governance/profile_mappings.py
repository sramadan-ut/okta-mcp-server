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
"""

import asyncio
import re
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
_DETAIL_CONCURRENCY = 4

_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?')


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


async def _get_all(ctx: Context, client, url: str, what: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch every page of a /api/v1 list. Returns (items, info)."""
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


async def _scan(ctx: Context, client, attributes: List[str], include_saml: bool) -> Dict[str, Any]:
    pattern = _attribute_pattern(attributes)
    wanted = set(attributes)
    matches: List[Dict[str, Any]] = []
    errors: List[str] = []
    pagination: Dict[str, Any] = {}

    # 1 + 2. Profile mappings in both directions between Okta users and apps.
    mappings, pagination["mappings"] = await _get_all(
        ctx, client, _mgmt_url("/mappings", {"limit": _PAGE_LIMIT}), "profile mappings"
    )
    candidates = [
        m
        for m in mappings
        if {(m.get("source") or {}).get("type"), (m.get("target") or {}).get("type")} == {"user", "appuser"}
    ]

    sem = asyncio.Semaphore(_DETAIL_CONCURRENCY)

    async def fetch_detail(summary: Dict[str, Any]) -> None:
        async with sem:
            _, detail, err = await okta_raw_request(client, "GET", _mgmt_url(f"/mappings/{summary.get('id')}"))
        if err or not isinstance(detail, dict):
            errors.append(f"mapping {summary.get('id')}: {_error_result(err)['error'] if err else 'empty body'}")
            return
        matches.extend(_mapping_matches(summary, detail, pattern, wanted))

    await asyncio.gather(*(fetch_detail(m) for m in candidates))

    # 3. SAML attribute statements.
    if include_saml:
        apps, pagination["apps"] = await _get_all(
            ctx, client, _mgmt_url("/apps", {"limit": _PAGE_LIMIT}), "applications"
        )
        for app in apps:
            matches.extend(_saml_matches(app, pattern))

    matches.sort(key=lambda r: (str(r.get("app") or "").lower(), r["via"], str(r.get("app_attribute") or "")))
    by_attribute = {a: sorted({str(r["app"]) for r in matches if a in r["okta_attributes"]}) for a in attributes}
    return {
        "attributes": attributes,
        "apps_by_attribute": by_attribute,
        "matches": matches,
        "total_matches": len(matches),
        "mappings_checked": len(candidates),
        "errors": errors,
        "pagination_info": pagination,
        "complete": not errors and not any(p.get("stopped_early") for p in pagination.values()),
    }


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
    If ``complete`` is False, some pages or mappings could not be read; see
    ``errors`` and ``pagination_info`` and say the result may be partial.

    Read-only. Makes one request per Okta<->app mapping, so it can take a
    while in large orgs.

    Parameters:
        attributes (list[str], required): Okta user profile variable names,
            case-sensitive, e.g. ["Team", "timeType"]. Check exact names
            with get_user_profile_attributes if unsure.
        include_saml (bool, optional): Also scan SAML attribute statements.
            Default: True.

    Returns:
        Dict containing:
        - apps_by_attribute: {attribute: [app names]}
        - matches: One row per app/attribute/expression with direction, via,
          app_attribute, expression and push_status
        - total_matches, mappings_checked
        - complete: False if any part of the scan failed or was truncated
        - errors, pagination_info
    """
    attributes = [a.strip() for a in (attributes or []) if a and a.strip()]
    if not attributes:
        return {"error": 'Provide at least one attribute name, e.g. ["Team", "timeType"].'}
    invalid = [a for a in attributes if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", a)]
    if invalid:
        return {"error": f"Invalid attribute names (use the variable name, not the display name): {invalid}"}

    logger.info(f"Finding profile attribute usage for {attributes} (include_saml={include_saml})")
    manager = ctx.request_context.lifespan_context.okta_auth_manager

    try:
        client = await get_okta_client(manager)
        result = await _scan(ctx, client, attributes, include_saml)
        logger.info(
            f"Profile attribute usage for {attributes}: {result['total_matches']} matches, "
            f"complete={result['complete']}"
        )
        return result
    except _ApiError as e:
        logger.error(f"Okta API error while finding profile attribute usage: {e.err}")
        return _error_result(e.err)
    except Exception as e:
        logger.error(f"Exception while finding profile attribute usage: {type(e).__name__}: {e}")
        return {"error": str(e)}

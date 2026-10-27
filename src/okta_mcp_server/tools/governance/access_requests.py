# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2026-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Read-only Okta Identity Governance (Access Requests) tools.

The governance API (``/governance/api/v2``) is not part of the management
SDK, so these tools go through ``okta_raw_request``.

Paging assumptions (not verifiable from the SDK; based on the published
governance API reference):

- List endpoints return ``{"data": [...], "_links": {"next": {"href": ...}}}``.
  A bare JSON array is also accepted and treated as a single, final page.
- More results exist only while ``_links.next.href`` is present. The href is
  followed as-is (reduced to path + query so the request always goes to the
  configured org), and its ``after`` query value is surfaced as
  ``next_cursor`` for manual paging.
- ``limit`` is forwarded unchanged; the server enforces its own bounds.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

from loguru import logger
from mcp.server.fastmcp import Context

from okta_mcp_server.server import mcp
from okta_mcp_server.utils.client import get_okta_client
from okta_mcp_server.utils.pagination import build_query_params, create_paginated_response
from okta_mcp_server.utils.raw_request import okta_raw_request
from okta_mcp_server.utils.scope_guard import require_scopes
from okta_mcp_server.utils.serialization import json_response, none_body_error
from okta_mcp_server.utils.validation import validate_ids

_GOVERNANCE_BASE = "/governance/api/v2"

#: Returns only top-level catalog entries (apps), not their child entries
#: (entitlements, bundles, groups).
_TOP_LEVEL_CATALOG_FILTER = "not(parent pr)"

#: Safety limits for fetch_all, mirroring paginate_all_results defaults.
_GOVERNANCE_MAX_PAGES = 500
_GOVERNANCE_PAGE_DELAY_SECONDS = 0.1


def _governance_url(path: str, params: Optional[Dict[str, Any]] = None) -> str:
    query_string = urlencode(params or {})
    return f"{_GOVERNANCE_BASE}{path}" + (f"?{query_string}" if query_string else "")


def _split_page(body: Any) -> Tuple[List[Any], Optional[str]]:
    """Return ``(items, next_href)`` for one governance list response."""
    if isinstance(body, list):
        return body, None
    if not isinstance(body, dict):
        return [], None
    next_link = (body.get("_links") or {}).get("next") or {}
    return body.get("data") or [], next_link.get("href") or None


def _next_request_url(href: Optional[str]) -> Optional[str]:
    """Reduce a ``_links.next.href`` to a governance path + query.

    Dropping scheme and host means the executor always prefixes the configured
    org URL, so the bearer token is never sent to a host taken from a response
    body. Hrefs outside the governance API are ignored.
    """
    if not href:
        return None
    parsed = urlparse(href)
    if not parsed.path.startswith(f"{_GOVERNANCE_BASE}/"):
        logger.warning(f"Ignoring governance next link outside {_GOVERNANCE_BASE}: {parsed.path}")
        return None
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


def _cursor_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    return parse_qs(urlparse(url).query).get("after", [None])[0]


async def _list_governance(ctx: Context, client, url: str, fetch_all: bool, what: str) -> dict:
    """Fetch one page (or every page when ``fetch_all``) of a governance list.

    Returns the standard ``create_paginated_response`` shape, with
    ``has_more``/``next_cursor`` derived from ``_links.next``. Error results
    are ``{"error": ...}``.
    """
    _, body, err = await okta_raw_request(client, "GET", url)
    if err:
        logger.error(f"Okta API error while listing {what}: {err}")
        return {"error": str(err)}

    items, next_href = _split_page(body)
    next_url = _next_request_url(next_href)

    if not fetch_all:
        result = create_paginated_response(items, None, fetch_all_used=False)
        result["has_more"] = next_url is not None
        result["next_cursor"] = _cursor_from_url(next_url)
        return result

    all_items = list(items)
    seen_urls = {url}
    pages_fetched = 1
    pagination_info: Dict[str, Any] = {"stopped_early": False, "stop_reason": None}

    while next_url and pages_fetched < _GOVERNANCE_MAX_PAGES:
        if next_url in seen_urls:
            pagination_info["stopped_early"] = True
            pagination_info["stop_reason"] = "Pagination cursor did not advance"
            break
        seen_urls.add(next_url)

        await asyncio.sleep(_GOVERNANCE_PAGE_DELAY_SECONDS)
        _, page_body, page_err = await okta_raw_request(client, "GET", next_url)
        if page_err:
            logger.warning(f"Error fetching {what} page {pages_fetched + 1}: {page_err}")
            pagination_info["stopped_early"] = True
            pagination_info["stop_reason"] = f"API error: {page_err}"
            break

        page_items, page_next_href = _split_page(page_body)
        all_items.extend(page_items)
        pages_fetched += 1
        try:
            await ctx.info(f"Fetching {what}... {len(all_items)} fetched so far ({pages_fetched} pages)")
        except Exception:
            pass
        next_url = _next_request_url(page_next_href)

    if next_url and pages_fetched >= _GOVERNANCE_MAX_PAGES:
        pagination_info["stopped_early"] = True
        pagination_info["stop_reason"] = f"Reached maximum page limit ({_GOVERNANCE_MAX_PAGES})"
        logger.warning(f"Stopped {what} pagination at {_GOVERNANCE_MAX_PAGES} pages limit")

    pagination_info["pages_fetched"] = pages_fetched
    pagination_info["total_items"] = len(all_items)
    result = create_paginated_response(all_items, None, fetch_all_used=True, pagination_info=pagination_info)
    if pagination_info["stopped_early"] and next_url:
        # Partial result — surface the resume point instead of implying
        # the walk completed.
        result["has_more"] = True
        result["next_cursor"] = _cursor_from_url(next_url)
    return result


def _status_counts(conditions: List[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for condition in conditions:
        status = condition.get("status") if isinstance(condition, dict) else None
        key = str(status).upper() if status else "UNKNOWN"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _matches_app_name(entry: Any, needle: str) -> bool:
    if not isinstance(entry, dict):
        return False
    return any(
        needle in str(entry.get(field) or "").lower() for field in ("name", "label", "displayName")
    )


# ---------------------------------------------------------------------------
# Request conditions
# ---------------------------------------------------------------------------


@mcp.tool()
@require_scopes("okta.accessRequests.condition.read")
@validate_ids("app_id", error_return_type="dict")
@json_response
async def list_app_request_conditions(
    ctx: Context,
    app_id: str,
    after: Optional[str] = None,
    limit: Optional[int] = None,
    fetch_all: bool = False,
) -> dict:
    """List the Access Requests request conditions configured on an app.

    Use this when a user can't see an app under "Request apps" in their Okta
    dashboard, can see it but can't request the access they expect, or you
    need to explain who may request an app and what they can request. A
    request condition decides which users (requester audience) can request
    which access (the app, its groups, or entitlements) and which approval
    sequence runs.

    How to read the result:
    - No conditions at all: the app is not requestable by anyone.
    - ``status == "ACTIVE"``: the condition is in effect. Check that the user
      is in its requester audience.
    - ``status == "INACTIVE"``: the condition exists but is switched off and
      grants nothing until an admin activates it.
    - ``status == "INVALID"``: the condition references something that no
      longer resolves (e.g. a deleted group, entitlement, or approval
      sequence) and is not applied. An admin must fix it in the Admin Console
      (Identity Governance → Access Requests).
    ``status_counts`` summarises the conditions by status.

    Read-only. Follow up with get_app_request_condition for a single
    condition, or list_catalog_entries to check the app is in the catalog.

    Parameters:
        app_id (str, required): The Okta application ID (e.g. 0oa...),
            from list_applications.
        after (str, optional): Pagination cursor (``next_cursor`` from a
            previous call).
        limit (int, optional): Number of conditions per page.
        fetch_all (bool, optional): If True, follow every ``_links.next``
            page. Default: False.

    Returns:
        Dict containing:
        - items: List of request condition objects
        - status_counts: Count of conditions per status (e.g. {"ACTIVE": 1, "INVALID": 1})
        - total_fetched: Number of conditions returned
        - has_more: Boolean indicating if more results are available
        - next_cursor: Cursor for the next page (if has_more is True)
        - fetch_all_used: Boolean indicating if fetch_all was used
        - pagination_info: Additional pagination metadata (when fetch_all=True)
    """
    logger.info(f"Listing request conditions for app {app_id} (fetch_all={fetch_all})")

    manager = ctx.request_context.lifespan_context.okta_auth_manager

    try:
        client = await get_okta_client(manager)
        url = _governance_url(f"/resources/{app_id}/request-conditions", build_query_params(after=after, limit=limit))
        result = await _list_governance(ctx, client, url, fetch_all, "request conditions")
        if "error" in result:
            return result

        result["status_counts"] = _status_counts(result["items"])
        logger.info(
            f"Retrieved {result['total_fetched']} request conditions for app {app_id}: {result['status_counts']}"
        )
        return result
    except Exception as e:
        logger.error(f"Exception while listing request conditions for app {app_id}: {type(e).__name__}: {e}")
        return {"error": str(e)}


@mcp.tool()
@require_scopes("okta.accessRequests.condition.read")
@validate_ids("app_id", "condition_id", error_return_type="dict")
@json_response
async def get_app_request_condition(ctx: Context, app_id: str, condition_id: str) -> dict:
    """Get one Access Requests request condition on an app.

    Use this after list_app_request_conditions to inspect a specific
    condition in full: its ``status`` (ACTIVE, INACTIVE, or INVALID), the
    requester audience, the access scope it grants (app, groups, or
    entitlements), and its approval sequence. Typical questions: "why can't
    this user request this app?", "who approves requests for X?", or "why
    is this condition marked INVALID?". An INVALID condition is not applied;
    look for a reference to a deleted group, entitlement, or approval
    sequence. Read-only.

    Parameters:
        app_id (str, required): The Okta application ID (e.g. 0oa...).
        condition_id (str, required): The request condition ID, from
            list_app_request_conditions.

    Returns:
        Dict with the request condition, or error information.
    """
    logger.info(f"Getting request condition {condition_id} for app {app_id}")

    manager = ctx.request_context.lifespan_context.okta_auth_manager

    try:
        client = await get_okta_client(manager)
        url = _governance_url(f"/resources/{app_id}/request-conditions/{condition_id}")
        _, condition, err = await okta_raw_request(client, "GET", url)
        if err:
            logger.error(f"Okta API error while getting request condition {condition_id}: {err}")
            return {"error": str(err)}

        if condition is None:
            return none_body_error(
                "get_app_request_condition",
                f"retrieving request condition {condition_id!r} for app {app_id!r}",
                "Verify the IDs with list_app_request_conditions().",
            )

        logger.info(f"Successfully retrieved request condition {condition_id} for app {app_id}")
        return condition
    except Exception as e:
        logger.error(f"Exception while getting request condition {condition_id}: {type(e).__name__}: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Access Requests catalog
# ---------------------------------------------------------------------------


@mcp.tool()
@require_scopes("okta.accessRequests.catalog.read")
@json_response
async def list_catalog_entries(
    ctx: Context,
    app_name: Optional[str] = None,
    after: Optional[str] = None,
    limit: Optional[int] = None,
    fetch_all: bool = False,
) -> dict:
    """List the top-level apps in the Access Requests catalog.

    Use this when a user can't see an app under "Request apps", or to check
    whether an app is requestable at all. If an app is missing from the
    catalog, nobody can request it, so the fix is in its Access Requests
    setup, not the user's assignments. If it is present but a specific user still can't see it, check its request
    conditions with list_app_request_conditions (the user may be outside
    every ACTIVE condition's audience, or the conditions may be INACTIVE or
    INVALID).

    This is the governance catalog, not the OIN app catalog — for installing
    integrations use list_catalog_apps instead. Only top-level entries are
    returned (filter ``not(parent pr)``); child entries such as entitlements
    are excluded. Read-only.

    Parameters:
        app_name (str, optional): Case-insensitive substring to match
            against each entry's name/label. Matching is done client-side
            over the whole catalog, so passing it always walks every page.
        after (str, optional): Pagination cursor (``next_cursor`` from a
            previous call). Ignored when ``app_name`` is given.
        limit (int, optional): Number of entries per page.
        fetch_all (bool, optional): If True, follow every ``_links.next``
            page. Default: False.

    Returns:
        Dict containing:
        - items: List of catalog entries
        - total_fetched: Number of entries returned
        - has_more: Boolean indicating if more results are available
        - next_cursor: Cursor for the next page (if has_more is True)
        - fetch_all_used: Boolean indicating if all pages were walked
        - pagination_info: Additional pagination metadata (when all pages were walked)
    """
    logger.info(f"Listing Access Requests catalog entries (app_name='{app_name}', fetch_all={fetch_all})")

    manager = ctx.request_context.lifespan_context.okta_auth_manager

    try:
        client = await get_okta_client(manager)
        needle = app_name.strip().lower() if app_name else ""
        # The name match is client-side, so a single page could hide the
        # app; walk the whole catalog from the start instead.
        walk_all = fetch_all or bool(needle)
        params = build_query_params(
            filter=_TOP_LEVEL_CATALOG_FILTER, after=None if needle else after, limit=limit
        )
        result = await _list_governance(
            ctx, client, _governance_url("/catalogs/default/entries", params), walk_all, "catalog entries"
        )
        if "error" in result:
            return result

        if needle:
            result["items"] = [entry for entry in result["items"] if _matches_app_name(entry, needle)]
            result["total_fetched"] = len(result["items"])

        logger.info(f"Retrieved {result['total_fetched']} Access Requests catalog entries")
        return result
    except Exception as e:
        logger.error(f"Exception while listing Access Requests catalog entries: {type(e).__name__}: {e}")
        return {"error": str(e)}

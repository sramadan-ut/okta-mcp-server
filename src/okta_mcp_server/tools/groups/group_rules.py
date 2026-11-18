# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2026-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Read-only tools for Okta group rules (``/api/v1/groups/rules``).

Like the profile mapping tools, these go through ``okta_raw_request`` on plain
JSON so the ``_embedded.groupIdToGroupName`` expansion is kept. Next-page
links are reduced to path + query and must stay under ``/api/v1/``.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from loguru import logger
from mcp.server.fastmcp import Context

from okta_mcp_server.server import mcp
from okta_mcp_server.tools.governance.profile_mappings import (
    _MAX_PAGES,
    _PAGE_DELAY_SECONDS,
    _mgmt_url,
    _next_from_link_header,
)
from okta_mcp_server.utils.client import get_okta_client
from okta_mcp_server.utils.raw_request import okta_raw_request
from okta_mcp_server.utils.scope_guard import MissingScopeError, require_scopes
from okta_mcp_server.utils.serialization import json_response
from okta_mcp_server.utils.validation import validate_ids

#: Okta's maximum page size for GET /api/v1/groups/rules.
_MAX_LIMIT = 200
_EXPAND = "groupIdToGroupName"
_STATUSES = ("ACTIVE", "INACTIVE")


def summarize_group_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Compact view of a group rule. Group names come from ``_embedded.groupIdToGroupName``."""
    conditions = rule.get("conditions") or {}
    expression = (conditions.get("expression") or {}).get("value")
    excluded = ((conditions.get("people") or {}).get("users") or {}).get("exclude") or []
    group_ids = ((rule.get("actions") or {}).get("assignUserToGroups") or {}).get("groupIds") or []
    names = (rule.get("_embedded") or {}).get("groupIdToGroupName") or {}
    return {
        "id": rule.get("id"),
        "name": rule.get("name"),
        "status": rule.get("status"),
        "expression": expression,
        "excluded_user_ids": list(excluded),
        "assigns_to": [{"id": gid, "name": names.get(gid)} for gid in group_ids],
        "created": rule.get("created"),
        "lastUpdated": rule.get("lastUpdated"),
    }


def _error_result(err: Any) -> Dict[str, Any]:
    if not isinstance(err, MissingScopeError):
        return {"error": str(err)}
    return {
        "error": str(err),
        "status": err.status,
        "errorCode": err.error_code,
        "errorSummary": err.error_summary,
    }


def _next_cursor(response: Any, page: List[Dict[str, Any]], limit: int) -> Optional[str]:
    """``after`` value for the next page, or None on the last page.

    Uses the Link header when the response exposes headers; otherwise falls
    back to the last rule id when the page is full.
    """
    has_headers, next_url = _next_from_link_header(response)
    if has_headers:
        if not next_url:
            return None
        return parse_qs(urlparse(next_url).query).get("after", [None])[0]
    if len(page) >= limit and page and page[-1].get("id"):
        return str(page[-1]["id"])
    return None


async def _fetch_page(
    client, search: Optional[str], limit: int, after: Optional[str]
) -> Tuple[List[Dict[str, Any]], Optional[str], Any]:
    url = _mgmt_url("/groups/rules", {"limit": limit, "after": after, "search": search or None, "expand": _EXPAND})
    response, body, err = await okta_raw_request(client, "GET", url)
    if err:
        return [], None, err
    page = [r for r in (body if isinstance(body, list) else []) if isinstance(r, dict)]
    return page, _next_cursor(response, page, limit), None


@mcp.tool()
@require_scopes("okta.groups.read")
@json_response
async def list_group_rules(
    ctx: Context,
    search: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = _MAX_LIMIT,
    after: Optional[str] = None,
    fetch_all: bool = False,
) -> dict:
    """List Okta group rules (rules that add users to groups based on an expression).

    Use this for questions like "which rules put people in group X?" or
    "which rules use user.department?". Read-only.

    Parameters:
        search (str, optional): Only rules whose name matches this text (Okta search).
        status (str, optional): ACTIVE or INACTIVE. Filtered after fetching, so a
            page can hold fewer than ``limit`` rules.
        limit (int, optional): Rules per page, 1-200. Default: 200.
        after (str, optional): ``next_cursor`` from the previous call.
        fetch_all (bool, optional): Fetch every page. Default: False.

    Returns:
        Dict containing:
        - items: [{id, name, status, expression, excluded_user_ids,
          assigns_to: [{id, name}], created, lastUpdated}]
        - total_fetched, has_more, next_cursor, fetch_all_used
        - pagination_info: {pages_fetched, stopped_early, stop_reason} when fetch_all=True
    """
    wanted_status = None
    if status:
        wanted_status = status.strip().upper()
        if wanted_status not in _STATUSES:
            return {"error": f"Invalid status {status!r}: use ACTIVE or INACTIVE."}
    try:
        limit = min(max(int(limit), 1), _MAX_LIMIT)
    except (TypeError, ValueError):
        limit = _MAX_LIMIT

    logger.info(
        f"Listing group rules (search={search!r}, status={wanted_status}, limit={limit}, "
        f"after={after!r}, fetch_all={fetch_all})"
    )
    manager = ctx.request_context.lifespan_context.okta_auth_manager

    try:
        client = await get_okta_client(manager)
        rules, cursor, err = await _fetch_page(client, search, limit, after)
        if err:
            logger.error(f"Okta API error while listing group rules: {err}")
            return _error_result(err)

        pagination_info = None
        if fetch_all:
            pagination_info = {"pages_fetched": 1, "stopped_early": False, "stop_reason": None}
            seen = {after}
            while cursor:
                if cursor in seen:
                    pagination_info.update(stopped_early=True, stop_reason="Pagination cursor did not advance")
                    break
                if pagination_info["pages_fetched"] >= _MAX_PAGES:
                    pagination_info.update(
                        stopped_early=True, stop_reason=f"Reached maximum page limit ({_MAX_PAGES})"
                    )
                    break
                seen.add(cursor)
                await asyncio.sleep(_PAGE_DELAY_SECONDS)
                page, next_cursor, err = await _fetch_page(client, search, limit, cursor)
                if err:
                    pagination_info.update(stopped_early=True, stop_reason=f"API error: {_error_result(err)['error']}")
                    break
                rules.extend(page)
                cursor = next_cursor
                pagination_info["pages_fetched"] += 1
                try:
                    await ctx.info(f"Fetching group rules... {len(rules)} so far")
                except Exception:
                    pass

        items = [summarize_group_rule(r) for r in rules if not wanted_status or r.get("status") == wanted_status]
        # After a complete fetch_all there is nothing more; if it stopped early, the cursor lets the caller resume.
        result: Dict[str, Any] = {
            "items": items,
            "total_fetched": len(items),
            "has_more": bool(cursor),
            "next_cursor": cursor,
            "fetch_all_used": fetch_all,
        }
        if pagination_info is not None:
            result["pagination_info"] = pagination_info
        logger.info(f"Listed {len(items)} group rules (has_more={result['has_more']})")
        return result
    except Exception as e:
        logger.error(f"Exception while listing group rules: {type(e).__name__}: {e}")
        return {"error": str(e)}


@mcp.tool()
@require_scopes("okta.groups.read")
@validate_ids("rule_id", error_return_type="dict")
@json_response
async def get_group_rule(ctx: Context, rule_id: str) -> dict:
    """Get one Okta group rule by ID, with the names of the groups it assigns. Read-only.

    Parameters:
        rule_id (str, required): The group rule ID (starts with 0pr), e.g. from list_group_rules.

    Returns:
        Dict with id, name, status, expression, excluded_user_ids,
        assigns_to [{id, name}], created, lastUpdated, and ``raw`` (the full rule JSON).
    """
    logger.info(f"Getting group rule {rule_id}")
    manager = ctx.request_context.lifespan_context.okta_auth_manager

    try:
        client = await get_okta_client(manager)
        _, body, err = await okta_raw_request(
            client, "GET", _mgmt_url(f"/groups/rules/{rule_id}", {"expand": _EXPAND})
        )
        if err:
            logger.error(f"Okta API error while getting group rule {rule_id}: {err}")
            return _error_result(err)
        if not isinstance(body, dict):
            return {
                "error": f"Okta returned an empty body for group rule {rule_id!r}. "
                "Verify the ID with list_group_rules()."
            }
        return {**summarize_group_rule(body), "raw": body}
    except Exception as e:
        logger.error(f"Exception while getting group rule {rule_id}: {type(e).__name__}: {e}")
        return {"error": str(e)}

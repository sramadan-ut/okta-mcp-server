# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2026-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Tests for the read-only governance (Access Requests) tools, the shared raw
request helper's 401/403 handling, and collision-free scope stub names."""

from __future__ import annotations

import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from okta_mcp_server.tools.applications.applications import list_catalog_apps
from okta_mcp_server.tools.governance.access_requests import (
    get_app_request_condition,
    list_app_request_conditions,
    list_catalog_entries,
)
from okta_mcp_server.utils.scope_guard import build_scope_error, get_active_required_scopes
from okta_mcp_server.utils.scope_registry import TOOL_SCOPE_REGISTRY
from okta_mcp_server.utils.scope_stubs import _scope_to_stub_name

GOV = "okta_mcp_server.tools.governance.access_requests"
APP_ID = "0oa1abc2def3GHI4j5d7"
CONDITION_SCOPE = "okta.accessRequests.condition.read"
CATALOG_SCOPE = "okta.accessRequests.catalog.read"


def _make_ctx():
    from tests.conftest import FakeLifespanContext, FakeOktaAuthManager

    request_context = MagicMock()
    request_context.lifespan_context = FakeLifespanContext(
        okta_auth_manager=FakeOktaAuthManager()
    )
    ctx = MagicMock()
    ctx.request_context = request_context
    ctx.info = AsyncMock()
    return ctx


class _HttpErr(Exception):
    """Mimics okta.errors OktaAPIError: carries ``.status``, ``.error_code``
    and ``.error_summary``. Pass ``error_code=None`` to mimic HTTPError,
    which has no Okta error body."""

    def __init__(
        self,
        status,
        error_code="E0000006",
        error_summary="You do not have permission to perform the requested action",
    ):
        self.status = status
        if error_code is not None:
            self.error_code = error_code
            self.error_summary = error_summary
        super().__init__(f"Okta HTTP {status} {error_code} {error_summary}")


def _assert_governance_denied(result, status, scope, error_code="E0000006",
                              error_summary="You do not have permission to perform the requested action"):
    assert result["status"] == status
    assert result["errorCode"] == error_code
    assert result["errorSummary"] == error_summary
    assert result["required_scopes"] == [scope]
    message = result["error"]
    if error_code:
        assert f"HTTP {status} {error_code}: {error_summary}" in message
    assert scope in message
    assert "missing OAuth scope" in message
    assert "OR a missing admin role" in message
    assert "Access Requests Administrator" in message


def _client_paged(pages):
    """Client whose executor returns one result per successive execute call.

    Each entry is a JSON-serialisable body, or a pre-built (response, body,
    err) tuple for error pages.
    """
    executor = MagicMock()
    executor.create_request = AsyncMock(return_value=({"method": "X"}, None))
    executor.execute = AsyncMock(
        side_effect=[p if isinstance(p, tuple) else (MagicMock(), json.dumps(p), None) for p in pages]
    )
    client = MagicMock()
    client.get_request_executor = MagicMock(return_value=executor)
    return client


def _requested_urls(client):
    executor = client.get_request_executor.return_value
    return [c.kwargs["url"] for c in executor.create_request.call_args_list]


def _page(items, next_href=None):
    links = {"self": {"href": "https://test.okta.com/governance/api/v2/x"}}
    if next_href:
        links["next"] = {"href": next_href}
    return {"data": items, "_links": links}


ACTIVE = {"id": "rc1", "name": "Engineers", "status": "ACTIVE"}
INACTIVE = {"id": "rc2", "name": "Contractors", "status": "INACTIVE"}
INVALID = {"id": "rc3", "name": "Deleted group", "status": "INVALID"}


# ---------------------------------------------------------------------------
# Request conditions
# ---------------------------------------------------------------------------


class TestListAppRequestConditions:
    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_active_inactive_invalid_with_status_counts(self, mock_get_client):
        client = _client_paged([_page([ACTIVE, INACTIVE, INVALID])])
        mock_get_client.return_value = client

        result = await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID)

        assert [c["status"] for c in result["items"]] == ["ACTIVE", "INACTIVE", "INVALID"]
        assert result["status_counts"] == {"ACTIVE": 1, "INACTIVE": 1, "INVALID": 1}
        assert result["has_more"] is False
        assert result["next_cursor"] is None
        assert _requested_urls(client) == [f"/governance/api/v2/resources/{APP_ID}/request-conditions"]

    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_no_conditions(self, mock_get_client):
        mock_get_client.return_value = _client_paged([_page([])])

        result = await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID)

        assert result["items"] == []
        assert result["status_counts"] == {}

    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_single_page_exposes_next_cursor(self, mock_get_client):
        next_href = f"https://test.okta.com/governance/api/v2/resources/{APP_ID}/request-conditions?after=rc1&limit=1"
        client = _client_paged([_page([ACTIVE], next_href)])
        mock_get_client.return_value = client

        result = await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID, limit=1)

        assert result["has_more"] is True
        assert result["next_cursor"] == "rc1"
        assert len(_requested_urls(client)) == 1
        assert parse_qs(urlparse(_requested_urls(client)[0]).query) == {"limit": ["1"]}

    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_after_is_forwarded(self, mock_get_client):
        client = _client_paged([_page([INVALID])])
        mock_get_client.return_value = client

        await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID, after="rc2")

        assert parse_qs(urlparse(_requested_urls(client)[0]).query) == {"after": ["rc2"]}

    @pytest.mark.asyncio
    @patch(f"{GOV}._GOVERNANCE_PAGE_DELAY_SECONDS", 0)
    @patch(f"{GOV}.get_okta_client")
    async def test_fetch_all_follows_links_next(self, mock_get_client):
        base = f"/governance/api/v2/resources/{APP_ID}/request-conditions"
        client = _client_paged([
            _page([ACTIVE], f"https://test.okta.com{base}?after=rc1"),
            _page([INACTIVE], f"https://test.okta.com{base}?after=rc2"),
            _page([INVALID]),
        ])
        mock_get_client.return_value = client

        result = await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID, fetch_all=True)

        assert [c["id"] for c in result["items"]] == ["rc1", "rc2", "rc3"]
        assert result["status_counts"] == {"ACTIVE": 1, "INACTIVE": 1, "INVALID": 1}
        assert result["fetch_all_used"] is True
        assert result["has_more"] is False
        assert result["pagination_info"]["pages_fetched"] == 3
        assert result["pagination_info"]["stopped_early"] is False
        # Next hrefs are reduced to path + query so the executor always targets the org.
        assert _requested_urls(client) == [base, f"{base}?after=rc1", f"{base}?after=rc2"]

    @pytest.mark.asyncio
    @patch(f"{GOV}._GOVERNANCE_PAGE_DELAY_SECONDS", 0)
    @patch(f"{GOV}.get_okta_client")
    async def test_foreign_host_in_next_link_is_dropped(self, mock_get_client):
        base = f"/governance/api/v2/resources/{APP_ID}/request-conditions"
        client = _client_paged([
            _page([ACTIVE], f"https://evil.example.com{base}?after=rc1"),
            _page([INACTIVE]),
        ])
        mock_get_client.return_value = client

        await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID, fetch_all=True)

        assert _requested_urls(client)[1] == f"{base}?after=rc1"

    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_next_link_outside_governance_is_ignored(self, mock_get_client):
        client = _client_paged([_page([ACTIVE], "https://test.okta.com/api/v1/users?after=x")])
        mock_get_client.return_value = client

        result = await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID, fetch_all=True)

        assert result["has_more"] is False
        assert len(_requested_urls(client)) == 1

    @pytest.mark.asyncio
    @patch(f"{GOV}._GOVERNANCE_PAGE_DELAY_SECONDS", 0)
    @patch(f"{GOV}.get_okta_client")
    async def test_fetch_all_stops_when_cursor_does_not_advance(self, mock_get_client):
        same = f"https://test.okta.com/governance/api/v2/resources/{APP_ID}/request-conditions?after=rc1"
        client = _client_paged([_page([ACTIVE], same), _page([INACTIVE], same)])
        mock_get_client.return_value = client

        result = await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID, fetch_all=True)

        assert result["pagination_info"]["stopped_early"] is True
        assert result["pagination_info"]["stop_reason"] == "Pagination cursor did not advance"
        assert result["has_more"] is True
        assert result["next_cursor"] == "rc1"
        assert len(_requested_urls(client)) == 2

    @pytest.mark.asyncio
    @patch(f"{GOV}._GOVERNANCE_PAGE_DELAY_SECONDS", 0)
    @patch(f"{GOV}.get_okta_client")
    async def test_fetch_all_error_mid_walk_keeps_partial_result(self, mock_get_client):
        base = f"/governance/api/v2/resources/{APP_ID}/request-conditions"
        client = _client_paged([
            _page([ACTIVE], f"https://test.okta.com{base}?after=rc1"),
            (None, None, _HttpErr(500)),
        ])
        mock_get_client.return_value = client

        result = await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID, fetch_all=True)

        assert [c["id"] for c in result["items"]] == ["rc1"]
        assert result["pagination_info"]["stopped_early"] is True
        assert result["has_more"] is True
        assert result["next_cursor"] == "rc1"

    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_bare_list_body_is_single_page(self, mock_get_client):
        mock_get_client.return_value = _client_paged([[ACTIVE, INVALID]])

        result = await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID)

        assert result["status_counts"] == {"ACTIVE": 1, "INVALID": 1}
        assert result["has_more"] is False

    @pytest.mark.asyncio
    async def test_rejects_path_traversal_app_id(self):
        result = await list_app_request_conditions(ctx=_make_ctx(), app_id="../users")
        assert "error" in result


class TestGetAppRequestCondition:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("condition", [ACTIVE, INACTIVE, INVALID], ids=lambda c: c["status"])
    @patch(f"{GOV}.get_okta_client")
    async def test_returns_condition_for_each_status(self, mock_get_client, condition):
        client = _client_paged([condition])
        mock_get_client.return_value = client

        result = await get_app_request_condition(ctx=_make_ctx(), app_id=APP_ID, condition_id=condition["id"])

        assert result == condition
        assert _requested_urls(client) == [
            f"/governance/api/v2/resources/{APP_ID}/request-conditions/{condition['id']}"
        ]

    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_empty_body_returns_error(self, mock_get_client):
        client = _client_paged([(MagicMock(), "", None)])
        mock_get_client.return_value = client

        result = await get_app_request_condition(ctx=_make_ctx(), app_id=APP_ID, condition_id="rc1")

        assert "empty response" in result["error"]

    @pytest.mark.asyncio
    async def test_rejects_bad_condition_id(self):
        result = await get_app_request_condition(ctx=_make_ctx(), app_id=APP_ID, condition_id="rc1/../x")
        assert "error" in result


# ---------------------------------------------------------------------------
# Catalog entries
# ---------------------------------------------------------------------------


SALESFORCE = {"id": "cen1", "name": "Salesforce", "requestable": True}
SLACK = {"id": "cen2", "name": "Slack", "requestable": True}
ZOOM = {"id": "cen3", "label": "Zoom", "requestable": False}


class TestListCatalogEntries:
    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_sends_top_level_filter(self, mock_get_client):
        client = _client_paged([_page([SALESFORCE, SLACK])])
        mock_get_client.return_value = client

        result = await list_catalog_entries(ctx=_make_ctx())

        assert result["total_fetched"] == 2
        url = _requested_urls(client)[0]
        assert urlparse(url).path == "/governance/api/v2/catalogs/default/entries"
        assert parse_qs(urlparse(url).query) == {"filter": ["not(parent pr)"]}

    @pytest.mark.asyncio
    @patch(f"{GOV}._GOVERNANCE_PAGE_DELAY_SECONDS", 0)
    @patch(f"{GOV}.get_okta_client")
    async def test_app_name_walks_all_pages_and_filters(self, mock_get_client):
        base = "/governance/api/v2/catalogs/default/entries"
        client = _client_paged([
            _page([SALESFORCE], f"https://test.okta.com{base}?filter=not%28parent+pr%29&after=cen1"),
            _page([SLACK, ZOOM]),
        ])
        mock_get_client.return_value = client

        result = await list_catalog_entries(ctx=_make_ctx(), app_name="  zo ", after="ignored")

        assert result["items"] == [ZOOM]
        assert result["total_fetched"] == 1
        assert result["fetch_all_used"] is True
        assert len(_requested_urls(client)) == 2
        assert "after" not in parse_qs(urlparse(_requested_urls(client)[0]).query)

    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_app_name_no_match(self, mock_get_client):
        mock_get_client.return_value = _client_paged([_page([SALESFORCE, SLACK])])

        result = await list_catalog_entries(ctx=_make_ctx(), app_name="workday")

        assert result["items"] == []
        assert result["total_fetched"] == 0


# ---------------------------------------------------------------------------
# 401/403 → canonical scope error
# ---------------------------------------------------------------------------


class TestGovernanceForbidden:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 403])
    @patch(f"{GOV}.get_okta_client")
    async def test_list_conditions_names_scope_and_role(self, mock_get_client, status):
        mock_get_client.return_value = _client_paged([(None, None, _HttpErr(status))])

        result = await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID)

        _assert_governance_denied(result, status, CONDITION_SCOPE)

    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_get_condition_forbidden(self, mock_get_client):
        mock_get_client.return_value = _client_paged([(None, None, _HttpErr(403))])

        result = await get_app_request_condition(ctx=_make_ctx(), app_id=APP_ID, condition_id="rc1")

        _assert_governance_denied(result, 403, CONDITION_SCOPE)

    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_catalog_forbidden(self, mock_get_client):
        mock_get_client.return_value = _client_paged([(None, None, _HttpErr(403))])

        result = await list_catalog_entries(ctx=_make_ctx())

        _assert_governance_denied(result, 403, CATALOG_SCOPE)

    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_real_sdk_okta_api_error(self, mock_get_client):
        from okta.errors.okta_api_error import OktaAPIError

        details = MagicMock(status=403, headers={})
        err = OktaAPIError(
            "https://test.okta.com/governance/api/v2/catalogs/default/entries",
            details,
            {"errorCode": "E0000006", "errorSummary": "You do not have permission to perform the requested action"},
        )
        mock_get_client.return_value = _client_paged([(None, None, err)])

        result = await list_catalog_entries(ctx=_make_ctx())

        _assert_governance_denied(result, 403, CATALOG_SCOPE)

    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_without_okta_error_body(self, mock_get_client):
        mock_get_client.return_value = _client_paged([(None, None, _HttpErr(401, error_code=None))])

        result = await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID)

        _assert_governance_denied(result, 401, CONDITION_SCOPE, error_code=None, error_summary=None)
        assert "(HTTP 401)" in result["error"]

    @pytest.mark.asyncio
    @patch(f"{GOV}._GOVERNANCE_PAGE_DELAY_SECONDS", 0)
    @patch(f"{GOV}.get_okta_client")
    async def test_forbidden_mid_walk_explains_in_stop_reason(self, mock_get_client):
        base = f"/governance/api/v2/resources/{APP_ID}/request-conditions"
        mock_get_client.return_value = _client_paged([
            _page([ACTIVE], f"https://test.okta.com{base}?after=rc1"),
            (None, None, _HttpErr(403)),
        ])

        result = await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID, fetch_all=True)

        assert [c["id"] for c in result["items"]] == ["rc1"]
        reason = result["pagination_info"]["stop_reason"]
        assert "E0000006" in reason and "Access Requests Administrator" in reason


class TestScopeErrorFromRawRequest:
    @pytest.mark.asyncio
    @patch("okta_mcp_server.tools.applications.applications.get_okta_client")
    async def test_oin_catalog_uses_shared_helper(self, mock_get_client):
        mock_get_client.return_value = _client_paged([(None, None, _HttpErr(403))])

        result = await list_catalog_apps(ctx=_make_ctx())

        assert result == build_scope_error(["okta.apps.read"])

    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_other_errors_pass_through(self, mock_get_client):
        mock_get_client.return_value = _client_paged([(None, None, _HttpErr(500))])

        result = await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID)

        assert result == {"error": str(_HttpErr(500))}

    @pytest.mark.asyncio
    async def test_pre_call_scope_check_matches(self):
        ctx = _make_ctx()
        ctx.request_context.lifespan_context.okta_auth_manager.scopes = "okta.apps.read"

        result = await list_catalog_entries(ctx=ctx)

        assert result == build_scope_error([CATALOG_SCOPE])

    @pytest.mark.asyncio
    @patch(f"{GOV}.get_okta_client")
    async def test_active_scopes_reset_after_call(self, mock_get_client):
        mock_get_client.return_value = _client_paged([_page([])])

        await list_app_request_conditions(ctx=_make_ctx(), app_id=APP_ID)

        assert get_active_required_scopes() == ()


# ---------------------------------------------------------------------------
# Registration and scope stubs
# ---------------------------------------------------------------------------


def test_tools_registered_with_scopes():
    assert TOOL_SCOPE_REGISTRY["list_app_request_conditions"] == CONDITION_SCOPE
    assert TOOL_SCOPE_REGISTRY["get_app_request_condition"] == CONDITION_SCOPE
    assert TOOL_SCOPE_REGISTRY["list_catalog_entries"] == CATALOG_SCOPE


def test_main_imports_governance_before_scope_stubs():
    from okta_mcp_server import server

    source = inspect.getsource(server.main)
    assert "tools.governance import access_requests" in source
    assert source.index("tools.governance import access_requests") < source.index("import scope_stubs")


def test_new_stub_names():
    assert _scope_to_stub_name(CONDITION_SCOPE) == "access_requests_condition_read_scope_info"
    assert _scope_to_stub_name(CATALOG_SCOPE) == "access_requests_catalog_read_scope_info"


@pytest.mark.parametrize(
    "scope, expected",
    [
        ("okta.users.read", "users_read_scope_info"),
        ("okta.users.manage", "users_manage_scope_info"),
        ("okta.groups.read", "groups_read_scope_info"),
        ("okta.apps.manage", "apps_manage_scope_info"),
        ("okta.policies.read", "policies_read_scope_info"),
        ("okta.deviceAssurance.read", "device_assurance_read_scope_info"),
        ("okta.deviceAssurance.manage", "device_assurance_manage_scope_info"),
        ("okta.logs.read", "logs_read_scope_info"),
        ("okta.brands.read", "brands_read_scope_info"),
        ("okta.templates.manage", "templates_manage_scope_info"),
        ("okta.domains.read", "domains_read_scope_info"),
        ("okta.emailDomains.read", "email_domains_read_scope_info"),
        ("okta.emailDomains.manage", "email_domains_manage_scope_info"),
    ],
)
def test_existing_stub_names_unchanged(scope, expected):
    assert _scope_to_stub_name(scope) == expected


def test_stub_names_unique_across_registry():
    scopes = set(TOOL_SCOPE_REGISTRY.values())
    assert len({_scope_to_stub_name(s) for s in scopes}) == len(scopes)

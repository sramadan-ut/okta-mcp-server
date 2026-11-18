# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2026-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Tests for the read-only group rule tools (no network)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import pytest

from okta_mcp_server.tools.groups import group_rules as gr
from okta_mcp_server.utils.scope_registry import TOOL_SCOPE_REGISTRY


def _make_ctx(scopes=None):
    from tests.conftest import FakeLifespanContext, FakeOktaAuthManager

    manager = FakeOktaAuthManager()
    if scopes is not None:
        manager.scopes = scopes
    ctx = MagicMock()
    ctx.request_context.lifespan_context = FakeLifespanContext(okta_auth_manager=manager)
    ctx.info = AsyncMock()
    return ctx


def _rule(i, status="ACTIVE", expr='user.department == "Eng"', groups=("00gA",)):
    return {
        "id": f"0pr{i:02d}",
        "name": f"rule {i}",
        "status": status,
        "created": "2026-01-01T00:00:00.000Z",
        "lastUpdated": "2026-02-01T00:00:00.000Z",
        "conditions": {
            "expression": {"type": "urn:okta:expression:1.0", "value": expr},
            "people": {"users": {"exclude": ["00uX"]}},
        },
        "actions": {"assignUserToGroups": {"groupIds": list(groups)}},
        "_embedded": {"groupIdToGroupName": {"00gA": "Engineering", "00gB": "Everyone in Eng"}},
    }


RULES = [_rule(i, status="ACTIVE" if i % 3 else "INACTIVE") for i in range(7)]


def paged_rules(calls, rules=RULES):
    """Fake okta_raw_request paging ``rules`` with after=<id> and no headers (page-size fallback)."""

    def _req(client, method, url):
        calls.append(url)
        q = parse_qs(urlparse(url).query)
        limit = int(q["limit"][0])
        after = q.get("after", [None])[0]
        start = 0 if after is None else [r["id"] for r in rules].index(after) + 1
        return None, rules[start : start + limit], None

    return AsyncMock(side_effect=_req)


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch):
    monkeypatch.setattr(gr, "get_okta_client", AsyncMock(return_value=object()))
    monkeypatch.setattr(gr, "_PAGE_DELAY_SECONDS", 0)


def test_summary_shape_and_group_name_expansion():
    rule = _rule(1, groups=("00gA", "00gB", "00gMissing"))
    assert gr.summarize_group_rule(rule) == {
        "id": "0pr01",
        "name": "rule 1",
        "status": "ACTIVE",
        "expression": 'user.department == "Eng"',
        "excluded_user_ids": ["00uX"],
        "assigns_to": [
            {"id": "00gA", "name": "Engineering"},
            {"id": "00gB", "name": "Everyone in Eng"},
            {"id": "00gMissing", "name": None},
        ],
        "created": "2026-01-01T00:00:00.000Z",
        "lastUpdated": "2026-02-01T00:00:00.000Z",
    }
    # Sparse rules don't crash.
    assert gr.summarize_group_rule({"id": "0prZ"})["assigns_to"] == []


@pytest.mark.asyncio
async def test_list_group_rules_requests_expand_and_search(monkeypatch):
    calls = []
    monkeypatch.setattr(gr, "okta_raw_request", paged_rules(calls))
    result = await gr.list_group_rules(_make_ctx(), search="rule", limit=500)

    q = parse_qs(urlparse(calls[0]).query)
    assert urlparse(calls[0]).path == "/api/v1/groups/rules"
    assert q["expand"] == ["groupIdToGroupName"]
    assert q["search"] == ["rule"]
    assert q["limit"] == ["200"]  # clamped
    assert result["total_fetched"] == 7
    assert result["has_more"] is False and result["next_cursor"] is None
    assert result["items"][0]["assigns_to"] == [{"id": "00gA", "name": "Engineering"}]


@pytest.mark.asyncio
async def test_list_group_rules_pagination_cursor_and_fetch_all(monkeypatch):
    calls = []
    monkeypatch.setattr(gr, "okta_raw_request", paged_rules(calls))

    first = await gr.list_group_rules(_make_ctx(), limit=3)
    assert [r["id"] for r in first["items"]] == ["0pr00", "0pr01", "0pr02"]
    assert first["has_more"] is True and first["next_cursor"] == "0pr02"

    second = await gr.list_group_rules(_make_ctx(), limit=3, after=first["next_cursor"])
    assert [r["id"] for r in second["items"]] == ["0pr03", "0pr04", "0pr05"]

    calls.clear()
    everything = await gr.list_group_rules(_make_ctx(), limit=3, fetch_all=True)
    assert [r["id"] for r in everything["items"]] == [r["id"] for r in RULES]
    assert everything["has_more"] is False and everything["next_cursor"] is None
    assert everything["pagination_info"]["pages_fetched"] == 3
    assert [parse_qs(urlparse(c).query).get("after", [None])[0] for c in calls] == [None, "0pr02", "0pr05"]


@pytest.mark.asyncio
async def test_list_group_rules_follows_link_header_only_under_api_v1(monkeypatch):
    from multidict import CIMultiDict, CIMultiDictProxy

    def resp(next_href):
        links = [("Link", '<https://org.okta.com/api/v1/groups/rules>; rel="self"')]
        if next_href:
            links.append(("Link", f'<{next_href}>; rel="next"'))
        return SimpleNamespace(headers=CIMultiDictProxy(CIMultiDict(links)))

    def _req(client, method, url):
        return resp("https://org.okta.com/api/v1/groups/rules?after=0pr09&limit=2"), RULES[:2], None

    monkeypatch.setattr(gr, "okta_raw_request", AsyncMock(side_effect=_req))
    result = await gr.list_group_rules(_make_ctx(), limit=2)
    assert result["next_cursor"] == "0pr09"

    def _evil(client, method, url):
        return resp("https://evil.example/api/v1/groups/rules?after=x"), RULES[:2], None

    monkeypatch.setattr(gr, "okta_raw_request", AsyncMock(side_effect=_evil))
    # The host is dropped and only the path + query is kept, so the cursor is still usable on the org.
    assert (await gr.list_group_rules(_make_ctx(), limit=2))["next_cursor"] == "x"

    def _outside(client, method, url):
        return resp("https://org.okta.com/oauth2/v1/whatever?after=x"), RULES[:2], None

    monkeypatch.setattr(gr, "okta_raw_request", AsyncMock(side_effect=_outside))
    assert (await gr.list_group_rules(_make_ctx(), limit=2))["next_cursor"] is None


@pytest.mark.asyncio
async def test_list_group_rules_status_filter(monkeypatch):
    monkeypatch.setattr(gr, "okta_raw_request", paged_rules([]))
    inactive = await gr.list_group_rules(_make_ctx(), status="inactive")
    assert {r["status"] for r in inactive["items"]} == {"INACTIVE"}
    assert [r["id"] for r in inactive["items"]] == ["0pr00", "0pr03", "0pr06"]

    active = await gr.list_group_rules(_make_ctx(), status="ACTIVE")
    assert len(active["items"]) == 4

    bad = await gr.list_group_rules(_make_ctx(), status="DELETED")
    assert "ACTIVE or INACTIVE" in bad["error"]


@pytest.mark.asyncio
async def test_get_group_rule_returns_shape_and_raw(monkeypatch):
    calls = []

    def _req(client, method, url):
        calls.append(url)
        return None, _rule(4), None

    monkeypatch.setattr(gr, "okta_raw_request", AsyncMock(side_effect=_req))
    result = await gr.get_group_rule(_make_ctx(), rule_id="0pr04")
    assert calls == ["/api/v1/groups/rules/0pr04?expand=groupIdToGroupName"]
    assert result["id"] == "0pr04"
    assert result["assigns_to"] == [{"id": "00gA", "name": "Engineering"}]
    assert result["raw"] == _rule(4)


@pytest.mark.asyncio
@pytest.mark.parametrize("rule_id", ["../users/00u1", "0pr1/lifecycle", "0pr1?x=1", "0pr%2F1", ""])
async def test_get_group_rule_rejects_invalid_id(monkeypatch, rule_id):
    request = AsyncMock()
    monkeypatch.setattr(gr, "okta_raw_request", request)
    result = await gr.get_group_rule(_make_ctx(), rule_id=rule_id)
    assert "error" in result
    request.assert_not_called()


@pytest.mark.asyncio
async def test_errors_are_returned_not_raised(monkeypatch):
    err = SimpleNamespace(status=404)
    monkeypatch.setattr(gr, "okta_raw_request", AsyncMock(return_value=(None, None, err)))
    assert "error" in await gr.get_group_rule(_make_ctx(), rule_id="0prNope")
    assert "error" in await gr.list_group_rules(_make_ctx())


@pytest.mark.asyncio
async def test_missing_scope_blocks_before_any_request(monkeypatch):
    request = AsyncMock()
    monkeypatch.setattr(gr, "okta_raw_request", request)
    result = await gr.list_group_rules(_make_ctx(scopes="okta.users.read"))
    assert "okta.groups.read" in result["error"]
    request.assert_not_called()


def test_registered_under_groups_read():
    assert TOOL_SCOPE_REGISTRY["list_group_rules"] == "okta.groups.read"
    assert TOOL_SCOPE_REGISTRY["get_group_rule"] == "okta.groups.read"

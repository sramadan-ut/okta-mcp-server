# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2026-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Tests for okta_mcp_server.tools.governance.profile_mappings (logic only, no network)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from okta_mcp_server.tools.governance import profile_mappings as pm


@pytest.fixture(autouse=True)
def _clear_caches():
    pm._clear_caches()
    yield
    pm._clear_caches()


MAPPINGS = [
    {
        "id": "prm1",
        "source": {"id": "oty1", "name": "user", "type": "user"},
        "target": {"id": "0oaA", "name": "Slack", "type": "appuser"},
    },
    {
        "id": "prm2",
        "source": {"id": "0oaB", "name": "BambooHR", "type": "appuser"},
        "target": {"id": "oty1", "name": "user", "type": "user"},
    },
    {
        "id": "prm3",
        "source": {"id": "oty1", "name": "user", "type": "user"},
        "target": {"id": "0oaC", "name": "Zoom", "type": "appuser"},
    },
]
DETAILS = {
    "prm1": {
        "properties": {
            "team": {"expression": "user.Team", "pushStatus": "PUSH"},
            "title": {"expression": "user.title", "pushStatus": "PUSH"},
        }
    },
    "prm2": {
        "properties": {
            "Team": {"expression": "appuser.department"},
            "timeType": {"expression": "appuser.timeType"},
            "title": {"expression": "appuser.title"},
        }
    },
    "prm3": {"properties": {"x": {"expression": "user.TeamLead"}}},  # must NOT match Team
}
APPS = [
    {
        "id": "0oaD",
        "label": "Expel.io",
        "status": "ACTIVE",
        "settings": {
            "signOn": {
                "attributeStatements": [{"type": "EXPRESSION", "name": "employment", "values": ["user.timeType"]}]
            }
        },
    },
    {"id": "0oaE", "label": "Zapier", "name": "zapier", "settings": {"app": {}}},
]


def fake_request():
    def _req(client, method, url):
        resp = SimpleNamespace(headers={})
        if url.startswith("/api/v1/mappings/"):
            return resp, DETAILS[url.rsplit("/", 1)[1]], None
        if url.startswith("/api/v1/mappings"):
            return resp, MAPPINGS, None
        if url.startswith("/api/v1/apps"):
            return resp, APPS, None
        raise AssertionError(url)

    return AsyncMock(side_effect=_req)


@pytest.mark.asyncio
async def test_scan_finds_reads_writes_and_saml(monkeypatch):
    monkeypatch.setattr(pm, "okta_raw_request", fake_request())
    ctx = SimpleNamespace(info=AsyncMock())
    result = await pm._scan(ctx, object(), ["Team", "timeType"], include_saml=True)

    assert result["complete"] is True
    assert result["apps_by_attribute"] == {"Team": ["BambooHR", "Slack"], "timeType": ["BambooHR", "Expel.io"]}
    vias = {(r["app"], r["direction"], r["via"]) for r in result["matches"]}
    assert ("Slack", "okta_to_app", "profile_mapping") in vias
    assert ("BambooHR", "app_to_okta", "profile_mapping") in vias
    assert ("Expel.io", "okta_to_app", "saml_attribute_statement") in vias
    assert not any(r["app"] == "Zoom" for r in result["matches"])


def test_pattern_is_exact_and_case_sensitive():
    p = pm._attribute_pattern(["Team"])
    assert p.search("user.Team")
    assert p.search('user.Team != null ? user.Team : ""')
    assert not p.search("user.TeamLead")
    assert not p.search("user.team")


def test_link_header_pagination_stays_on_org():
    resp = SimpleNamespace(headers={"link": '<https://org.okta.com/api/v1/mappings?after=prm9&limit=200>; rel="next"'})
    assert pm._next_from_link_header(resp) == (True, "/api/v1/mappings?after=prm9&limit=200")
    evil = SimpleNamespace(headers={"link": '<https://evil.example/steal>; rel="next"'})
    assert pm._next_from_link_header(evil) == (True, None)


@pytest.mark.asyncio
async def test_after_fallback_when_no_headers(monkeypatch):
    calls = []
    full_page = [{"id": f"x{i}"} for i in range(pm._PAGE_LIMIT)]

    def _req(client, method, url):
        calls.append(url)
        return None, (full_page if len(calls) == 1 else [{"id": "last"}]), None

    monkeypatch.setattr(pm, "okta_raw_request", AsyncMock(side_effect=_req))
    items, _info = await pm._get_all(SimpleNamespace(info=AsyncMock()), object(), "/api/v1/apps?limit=200", "apps")
    assert len(items) == pm._PAGE_LIMIT + 1
    assert "after=x199" in calls[1]


def test_link_header_from_aiohttp_multidict():
    """okta_raw_request returns aiohttp.ClientResponse; Okta sends self + next as separate Link headers."""
    from multidict import CIMultiDict, CIMultiDictProxy

    headers = CIMultiDictProxy(
        CIMultiDict(
            [
                ("Link", '<https://org.okta.com/api/v1/apps?limit=200>; rel="self"'),
                ("Link", '<https://org.okta.com/api/v1/apps?after=0oa9&limit=200>; rel="next"'),
            ]
        )
    )
    assert pm._next_from_link_header(SimpleNamespace(headers=headers)) == (True, "/api/v1/apps?after=0oa9&limit=200")


@pytest.mark.asyncio
async def test_last_page_with_headers_does_not_use_after_fallback(monkeypatch):
    from multidict import CIMultiDict, CIMultiDictProxy

    calls = []
    full_page = [{"id": f"x{i}"} for i in range(pm._PAGE_LIMIT)]
    only_self = CIMultiDictProxy(CIMultiDict([("Link", '<https://org.okta.com/api/v1/apps>; rel="self"')]))

    def _req(client, method, url):
        calls.append(url)
        return SimpleNamespace(headers=only_self), full_page, None

    monkeypatch.setattr(pm, "okta_raw_request", AsyncMock(side_effect=_req))
    _items, info = await pm._get_all(SimpleNamespace(info=AsyncMock()), object(), "/api/v1/apps?limit=200", "apps")
    assert len(calls) == 1
    assert info["stopped_early"] is False


# ---------------------------------------------------------------------------
# Resumability: time budget, cache, 429, app_name
# ---------------------------------------------------------------------------

MANY_MAPPINGS = [
    {
        "id": f"prm{i}",
        "source": {"id": "oty1", "name": "user", "type": "user"},
        "target": {"id": f"0oa{i}", "name": f"app{i}", "type": "appuser"},
    }
    for i in range(40)
]


def slow_request(calls, detail_delay=0.05):
    """Fake okta_raw_request: every detail reads user.Team and takes ``detail_delay`` seconds."""

    async def _req(client, method, url):
        calls.append(url)
        resp = SimpleNamespace(headers={})
        if url.startswith("/api/v1/mappings/"):
            await asyncio.sleep(detail_delay)
            return resp, {"properties": {"team": {"expression": "user.Team"}}}, None
        if url.startswith("/api/v1/mappings"):
            return resp, MANY_MAPPINGS, None
        if url.startswith("/api/v1/apps"):
            return resp, APPS, None
        raise AssertionError(url)

    return _req


def _detail_calls(calls):
    return [c for c in calls if c.startswith("/api/v1/mappings/")]


@pytest.mark.asyncio
async def test_budget_expiry_returns_partial_then_second_call_completes_from_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(pm, "okta_raw_request", slow_request(calls))
    ctx = SimpleNamespace(info=AsyncMock())

    # 40 details x 0.05s / 6 workers ~ 0.35s, so a 0.12s budget cannot finish.
    first = await pm._scan(ctx, object(), ["Team"], max_seconds=0.12)
    assert first["complete"] is False
    assert first["mappings_remaining"] > 0
    assert first["mappings_checked"] + first["mappings_remaining"] == len(MANY_MAPPINGS)
    assert "call find_profile_attribute_usage again with the same arguments" in first["message"]
    assert first["errors"] == []
    # Matches cover everything fetched so far.
    assert len(first["apps_by_attribute"]["Team"]) == first["mappings_checked"]
    first_detail_calls = len(_detail_calls(calls))

    calls.clear()
    second = await pm._scan(ctx, object(), ["Team"], max_seconds=10)
    assert second["complete"] is True
    assert second["mappings_remaining"] == 0
    assert second["mappings_from_cache"] == first["mappings_checked"]
    assert "message" not in second
    assert len(second["apps_by_attribute"]["Team"]) == len(MANY_MAPPINGS)
    # Only uncached details were fetched; the apps list came from the cache.
    assert len(_detail_calls(calls)) == first["mappings_remaining"]
    assert first_detail_calls + len(_detail_calls(calls)) <= len(MANY_MAPPINGS) + pm._DETAIL_CONCURRENCY
    assert not any(c.startswith("/api/v1/apps") for c in calls)
    assert second["saml_apps_from_cache"] is True


@pytest.mark.asyncio
async def test_cache_expires_after_ttl(monkeypatch):
    calls = []
    monkeypatch.setattr(pm, "okta_raw_request", slow_request(calls, detail_delay=0))
    ctx = SimpleNamespace(info=AsyncMock())
    await pm._scan(ctx, object(), ["Team"])
    for key, (ts, detail) in list(pm._DETAIL_CACHE.items()):
        pm._DETAIL_CACHE[key] = (ts - pm._CACHE_TTL_SECONDS - 1, detail)
    pm._APPS_CACHE["fetched_at"] -= pm._CACHE_TTL_SECONDS + 1

    calls.clear()
    result = await pm._scan(ctx, object(), ["Team"])
    assert result["mappings_from_cache"] == 0
    assert len(_detail_calls(calls)) == len(MANY_MAPPINGS)
    assert any(c.startswith("/api/v1/apps") for c in calls)


@pytest.mark.asyncio
async def test_429_stops_fetching_and_returns_partial(monkeypatch):
    calls = []
    rate_limited = SimpleNamespace(status=429, error_code="E0000047", error_summary="API call exceeded rate limit")

    def _req(client, method, url):
        calls.append(url)
        resp = SimpleNamespace(headers={})
        if url.startswith("/api/v1/mappings/"):
            return None, None, rate_limited
        if url.startswith("/api/v1/mappings"):
            return resp, MANY_MAPPINGS, None
        return resp, APPS, None

    monkeypatch.setattr(pm, "okta_raw_request", AsyncMock(side_effect=_req))
    result = await pm._scan(SimpleNamespace(info=AsyncMock()), object(), ["Team", "timeType"])

    assert result["complete"] is False
    assert len(_detail_calls(calls)) <= pm._DETAIL_CONCURRENCY
    assert result["mappings_remaining"] == len(MANY_MAPPINGS)
    assert len([e for e in result["errors"] if "429" in e]) == 1
    assert "rate-limited" in result["message"]
    # SAML results are still returned.
    assert result["apps_by_attribute"]["timeType"] == ["Expel.io"]


def test_is_rate_limited_variants():
    assert pm._is_rate_limited(SimpleNamespace(status=429))
    assert pm._is_rate_limited(Exception("429 response must have the 'X-Rate-Limit-Reset' and 'Date' headers"))
    assert not pm._is_rate_limited(SimpleNamespace(status=404))


@pytest.mark.asyncio
async def test_app_name_filters_mappings_and_saml_before_fetching(monkeypatch):
    request = fake_request()
    monkeypatch.setattr(pm, "okta_raw_request", request)
    ctx = SimpleNamespace(info=AsyncMock())

    slack = await pm._scan(ctx, object(), ["Team", "timeType"], app_name="SLA")
    urls = [c.args[2] for c in request.call_args_list]
    assert [u for u in urls if u.startswith("/api/v1/mappings/")] == ["/api/v1/mappings/prm1"]
    assert slack["mappings_total"] == 1
    assert {r["app"] for r in slack["matches"]} == {"Slack"}
    assert slack["complete"] is True

    request.reset_mock()
    expel = await pm._scan(ctx, object(), ["Team", "timeType"], app_name="expel")
    urls = [c.args[2] for c in request.call_args_list]
    assert not any(u.startswith("/api/v1/mappings/") for u in urls)
    assert {r["via"] for r in expel["matches"]} == {"saml_attribute_statement"}
    assert expel["apps_by_attribute"] == {"Team": [], "timeType": ["Expel.io"]}

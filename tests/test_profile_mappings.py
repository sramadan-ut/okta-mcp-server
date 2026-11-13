# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2026-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Tests for okta_mcp_server.tools.governance.profile_mappings (logic only, no network)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from okta_mcp_server.tools.governance import profile_mappings as pm

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

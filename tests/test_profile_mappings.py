# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2026-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Tests for okta_mcp_server.tools.governance.profile_mappings (logic only, no network)."""

import asyncio
import json
import os
import stat
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest

from okta_mcp_server.tools.governance import profile_mappings as pm
from okta_mcp_server.utils.scope_guard import MissingScopeError


@pytest.fixture(autouse=True)
def _isolated_caches(monkeypatch, tmp_path):
    pm._clear_caches()
    monkeypatch.setattr(pm, "_DISK_CACHE_DIR", tmp_path)
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
# Cursor-based resumption: phases, chunks, 403s, 429s, app_name, disk cache
# ---------------------------------------------------------------------------

MANY_MAPPINGS = [
    {
        "id": f"prm{i:02d}",
        "source": {"id": "oty1", "name": "user", "type": "user"},
        "target": {"id": f"0oa{i:02d}", "name": f"app{i}", "type": "appuser"},
    }
    for i in range(40)
]
MANY_APPS = [
    {
        "id": f"0oaS{i:02d}",
        "label": f"saml{i}",
        "settings": {
            "signOn": {
                "attributeStatements": [{"name": "team", "values": ["user.Team" if i % 2 == 0 else "user.department"]}]
            }
        },
    }
    for i in range(7)
]
ALL_MAPPING_IDS = {m["id"] for m in MANY_MAPPINGS}
SAML_TEAM_APPS = {"saml0", "saml2", "saml4", "saml6"}


def paged_request(calls, detail_delay=0.0, page_delay=0.0, detail_errors=None):
    """Fake okta_raw_request: 40 mappings (listed out of order) that all read user.Team, and
    MANY_APPS paged with after=<id> and no headers, so paging falls back to page size."""

    async def _req(client, method, url):
        calls.append(url)
        if url.startswith("/api/v1/mappings/"):
            await asyncio.sleep(detail_delay)
            mapping_id = url.rsplit("/", 1)[1]
            error = (detail_errors or {}).get(mapping_id)
            if callable(error):
                error = error()
            if error is not None:
                return None, None, error
            return SimpleNamespace(headers={}), {"properties": {"team": {"expression": "user.Team"}}}, None
        if url.startswith("/api/v1/mappings"):
            return SimpleNamespace(headers={}), list(reversed(MANY_MAPPINGS)), None
        if url.startswith("/api/v1/apps"):
            await asyncio.sleep(page_delay)
            after = parse_qs(urlparse(url).query).get("after", [None])[0]
            start = 0 if after is None else [a["id"] for a in MANY_APPS].index(after) + 1
            return None, MANY_APPS[start : start + pm._PAGE_LIMIT], None
        raise AssertionError(url)

    return _req


def _detail_calls(calls):
    return [c for c in calls if c.startswith("/api/v1/mappings/")]


async def run_to_completion(attributes, max_seconds, fresh_state=True, **kwargs):
    """Call _scan with each next_cursor until it is None, like a client would."""
    ctx = SimpleNamespace(info=AsyncMock())
    results, cursor = [], None
    for _ in range(200):
        if fresh_state:
            pm._clear_caches()  # a fresh server process per call
        result = await pm._scan(ctx, object(), attributes, max_seconds=max_seconds, cursor=cursor, **kwargs)
        results.append(result)
        cursor = result["next_cursor"]
        if cursor is None:
            return results
    raise AssertionError("scan did not finish")


def _mapping_ids(results):
    return [r["mapping_id"] for res in results for r in res["matches"] if r["via"] == "profile_mapping"]


def _saml_apps(results):
    return [r["app"] for res in results for r in res["matches"] if r["via"] == "saml_attribute_statement"]


@pytest.mark.asyncio
async def test_cursor_resumes_in_fresh_state_through_all_phases(monkeypatch):
    calls = []
    monkeypatch.setattr(pm, "_PAGE_LIMIT", 3)
    monkeypatch.setattr(pm, "okta_raw_request", paged_request(calls, detail_delay=0.02, page_delay=0.03))

    results = await run_to_completion(["Team"], max_seconds=0.05)

    order = {"mappings": 0, "saml": 1, "done": 2}
    phases = [r["progress"]["phase"] for r in results]
    assert phases == sorted(phases, key=order.get)
    assert phases[0] == "mappings" and "saml" in phases and phases[-1] == "done"
    assert len(results) >= 4
    # Every chunk but the last points to the next one.
    assert all(r["next_cursor"] and r["complete"] is False and "message" in r for r in results[:-1])
    assert results[-1]["complete"] is True and "message" not in results[-1]

    # Each mapping and SAML app is reported exactly once across chunks.
    mapping_ids = _mapping_ids(results)
    assert len(mapping_ids) == len(set(mapping_ids))
    assert set(mapping_ids) == ALL_MAPPING_IDS
    saml_apps = _saml_apps(results)
    assert len(saml_apps) == len(set(saml_apps))
    assert set(saml_apps) == SAML_TEAM_APPS

    # Nothing came from a cache, so correctness doesn't depend on it.
    assert all(r["mappings_from_cache"] == 0 for r in results)
    # Mapping progress counts up to the total in id order.
    mapping_progress = [r["progress"] for r in results if r["progress"]["phase"] == "mappings"]
    assert all(p["total"] == len(MANY_MAPPINGS) for p in mapping_progress)
    assert [p["checked_so_far"] for p in mapping_progress] == sorted(p["checked_so_far"] for p in mapping_progress)
    # SAML pages are requested with after=<last app id> and never go backwards. (A page whose
    # request was cancelled at the deadline is requested again by the next call.)
    afters = [parse_qs(urlparse(c).query).get("after", [""])[0] for c in calls if c.startswith("/api/v1/apps")]
    assert afters == sorted(afters)
    assert len(set(afters)) == 3  # 7 apps / 3 per page


@pytest.mark.asyncio
async def test_include_saml_false_goes_straight_to_done(monkeypatch):
    calls = []
    monkeypatch.setattr(pm, "okta_raw_request", paged_request(calls))
    results = await run_to_completion(["Team"], max_seconds=10, include_saml=False)
    assert [r["progress"]["phase"] for r in results] == ["done"]
    assert results[0]["complete"] is True
    assert not any(c.startswith("/api/v1/apps") for c in calls)


@pytest.mark.asyncio
async def test_403s_are_listed_and_do_not_block_completion(monkeypatch):
    calls = []
    forbidden = {
        "prm03": SimpleNamespace(status=403, error_code="E0000006"),
        "prm17": MissingScopeError(["okta.profileMappings.read"], status=403, error_code="E0000006"),
    }
    monkeypatch.setattr(pm, "okta_raw_request", paged_request(calls, detail_errors=forbidden))

    results = await run_to_completion(["Team"], max_seconds=10)
    assert results[-1]["complete"] is True
    assert results[-1]["next_cursor"] is None
    listed = [f for r in results for f in r["forbidden_mappings"]]
    assert listed == [{"mapping_id": "prm03", "app": "app3"}, {"mapping_id": "prm17", "app": "app17"}]
    assert all(r["errors"] == [] for r in results)
    assert set(_mapping_ids(results)) == ALL_MAPPING_IDS - {"prm03", "prm17"}


@pytest.mark.asyncio
async def test_other_failures_are_carried_in_cursor_and_block_complete(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pm, "okta_raw_request", paged_request(calls, detail_delay=0.02, detail_errors={"prm01": Exception("boom")})
    )
    results = await run_to_completion(["Team"], max_seconds=0.05, include_saml=False)
    assert len(results) > 1
    assert any("mapping prm01 (app1): boom" in e for e in results[0]["errors"])
    final = results[-1]
    assert final["next_cursor"] is None
    assert final["complete"] is False
    assert final["progress"]["failed_so_far"] == 1
    assert "could not be read" in final["message"]


@pytest.mark.asyncio
async def test_429_stops_early_and_resume_finishes(monkeypatch):
    calls = []
    hits = {"n": 0}

    def rate_limit_once():
        hits["n"] += 1
        return SimpleNamespace(status=429, error_code="E0000047") if hits["n"] == 1 else None

    monkeypatch.setattr(
        pm, "okta_raw_request", paged_request(calls, detail_delay=0.001, detail_errors={"prm10": rate_limit_once})
    )
    ctx = SimpleNamespace(info=AsyncMock())

    first = await pm._scan(ctx, object(), ["Team"], max_seconds=10)
    assert first["next_cursor"] is not None
    assert first["complete"] is False
    assert first["progress"]["phase"] == "mappings"
    assert first["progress"]["checked_so_far"] == 10  # prm00..prm09; prm10 is retried next call
    assert any("429" in e for e in first["errors"])
    assert "Wait about a minute" in first["message"]
    assert len(_detail_calls(calls)) < len(MANY_MAPPINGS)
    assert not any(c.startswith("/api/v1/apps") for c in calls)

    pm._clear_caches()  # fresh process
    resumed = await pm._scan(ctx, object(), ["Team"], max_seconds=10, cursor=first["next_cursor"])
    assert resumed["complete"] is True
    combined = _mapping_ids([first, resumed])
    assert len(combined) == len(set(combined))
    assert set(combined) == ALL_MAPPING_IDS


def test_is_rate_limited_variants():
    assert pm._is_rate_limited(SimpleNamespace(status=429))
    assert pm._is_rate_limited(Exception("429 response must have the 'X-Rate-Limit-Reset' and 'Date' headers"))
    assert not pm._is_rate_limited(SimpleNamespace(status=404))


@pytest.mark.asyncio
async def test_cursor_must_match_arguments_and_be_well_formed(monkeypatch):
    monkeypatch.setattr(pm, "okta_raw_request", paged_request([], detail_delay=0.02))
    ctx = SimpleNamespace(info=AsyncMock())
    first = await pm._scan(ctx, object(), ["Team"], max_seconds=0.05)
    assert first["next_cursor"]

    with pytest.raises(pm._CursorError, match="different arguments"):
        await pm._scan(ctx, object(), ["timeType"], cursor=first["next_cursor"])
    with pytest.raises(pm._CursorError, match="different arguments"):
        await pm._scan(ctx, object(), ["Team"], app_name="slack", cursor=first["next_cursor"])
    with pytest.raises(pm._CursorError, match="Invalid cursor"):
        await pm._scan(ctx, object(), ["Team"], cursor="not-a-cursor!!")
    with pytest.raises(pm._CursorError, match="Invalid cursor"):
        await pm._scan(ctx, object(), ["Team"], cursor=pm._encode_cursor({"v": 1, "phase": "bogus"}))


@pytest.mark.asyncio
async def test_app_name_filters_mappings_and_saml_before_fetching(monkeypatch):
    request = fake_request()
    monkeypatch.setattr(pm, "okta_raw_request", request)
    ctx = SimpleNamespace(info=AsyncMock())

    slack = await pm._scan(ctx, object(), ["Team", "timeType"], app_name="SLA")
    urls = [c.args[2] for c in request.call_args_list]
    assert [u for u in urls if u.startswith("/api/v1/mappings/")] == ["/api/v1/mappings/prm1"]
    assert {r["app"] for r in slack["matches"]} == {"Slack"}
    assert slack["complete"] is True

    request.reset_mock()
    expel = await pm._scan(ctx, object(), ["Team", "timeType"], app_name="expel")
    urls = [c.args[2] for c in request.call_args_list]
    assert not any(u.startswith("/api/v1/mappings/") for u in urls)
    assert {r["via"] for r in expel["matches"]} == {"saml_attribute_statement"}
    assert expel["apps_by_attribute"] == {"Team": [], "timeType": ["Expel.io"]}


@pytest.mark.asyncio
async def test_disk_cache_is_a_per_org_speedup_only(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pm, "okta_raw_request", paged_request(calls))
    org = "https://example.okta.com"

    first = await run_to_completion(["Team"], max_seconds=10, org_url=org)
    path = pm._disk_cache_path(org)
    assert path.parent == tmp_path and path.exists()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert pm._disk_cache_path("https://other.okta.com") != path

    # Fresh process, warm disk cache: no detail requests, same answer.
    calls.clear()
    warm = await run_to_completion(["Team"], max_seconds=10, org_url=org)
    assert _detail_calls(calls) == []
    assert warm[0]["mappings_from_cache"] == len(MANY_MAPPINGS)
    assert sorted(_mapping_ids(warm)) == sorted(_mapping_ids(first))

    # Expired entries are ignored.
    data = json.loads(path.read_text())
    for entry in data["entries"].values():
        entry[0] -= pm._CACHE_TTL_SECONDS + 1
    path.write_text(json.dumps(data))
    calls.clear()
    await run_to_completion(["Team"], max_seconds=10, org_url=org)
    assert len(_detail_calls(calls)) == len(MANY_MAPPINGS)

    # A corrupt or unwritable cache never breaks the scan.
    path.write_text("{not json")
    monkeypatch.setattr(pm, "_DISK_CACHE_DIR", tmp_path / "missing-dir")
    calls.clear()
    results = await run_to_completion(["Team"], max_seconds=10, org_url=org)
    assert results[-1]["complete"] is True
    assert set(_mapping_ids(results)) == ALL_MAPPING_IDS

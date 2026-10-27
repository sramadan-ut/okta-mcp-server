# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2026-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Raw request helper for Okta endpoints the typed SDK can't serve.

Used by the OIN catalog tools (no typed catalog surface) and the governance
tools (``/governance/api/v2`` is not in the management SDK).
"""

import json
from typing import Any, Dict, Optional

from loguru import logger

from okta_mcp_server.utils.scope_guard import MissingScopeError, get_active_required_scopes

#: HTTP statuses that mean the token lacks the scope (or the admin role) for the call.
_SCOPE_DENIED_STATUSES = (401, 403)


async def okta_raw_request(
    client, method: str, url: str, body: Optional[Dict[str, Any]] = None, keep_empty_params: bool = False
) -> tuple:
    """Issue a request through the SDK request executor and decode the JSON body.

    Args:
        keep_empty_params: Pass True to forward the body verbatim — by default
            the SDK's create_request prunes empty-string/list/dict values from
            the body (``clear_empty_params``).

    Returns:
        Tuple of ``(response, parsed_body, error)`` — the transport response,
        the JSON-decoded body (``None`` when the body is empty), and the SDK
        error (``None`` on success). ``response`` and ``parsed_body`` are
        ``None`` whenever ``error`` is set.

        When the executor reports HTTP 401/403 inside a ``@require_scopes``
        tool, ``error`` is a ``MissingScopeError`` for that tool's scopes, so
        ``{"error": str(error)}`` matches the decorator's own scope error.
    """
    executor = client.get_request_executor()
    request, err = await executor.create_request(
        method=method, url=url, body=body or {}, headers={}, oauth=False, keep_empty_params=keep_empty_params
    )
    if err:
        return None, None, err
    response, response_body, err = await executor.execute(request)
    if err:
        return None, None, _as_scope_error(err)
    return response, json.loads(response_body) if response_body else None, None


def _as_scope_error(err: Any) -> Any:
    status = getattr(err, "status", None)
    scopes = get_active_required_scopes()
    if status in _SCOPE_DENIED_STATUSES and scopes:
        logger.error(f"Okta API returned HTTP {status} — likely missing scope(s): {list(scopes)}")
        return MissingScopeError(list(scopes), status=status)
    return err

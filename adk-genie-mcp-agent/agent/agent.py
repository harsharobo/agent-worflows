"""
Databricks AI/BI Genie ADK Agent

Connects to a Databricks Genie Space via MCP (Streamable HTTP) and exposes
its tools to a Google ADK LlmAgent backed by the databricks-gemini-2-5-flash
model served from Databricks Foundation Model API.

Deployment target: Vertex AI Agent Engine (GCP)

Design notes:
  - Uses BaseAgent + set_up() pattern so MCPToolset is created AFTER unpickling
    on the Agent Engine container, avoiding the serialization issue with thread
    locks and open connections in MCPToolset.
  - The Genie MCP endpoint is:
      https://<DATABRICKS_HOST>/api/2.0/mcp/genie/<GENIE_SPACE_ID>
  - Model: gemini-2.5-flash served via Databricks Foundation Model API
    using the OpenAI-compatible endpoint with LiteLLM proxy format.
  - Token resolution order:
      1. DATABRICKS_TOKEN env var (static PAT) — passed as a fixed header
      2. OAuth M2M client credentials (DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET)
         — passed via McpToolset.header_provider so a fresh token is fetched
           (from the module-level cache) on every MCP session creation
      3. Databricks SDK profile fallback (~/.databrickscfg) — static header

Local dev:
    adk web agent
"""

import os
import time
import urllib.parse
import urllib.request
import json
from typing import AsyncGenerator, Optional, Tuple

from dotenv import load_dotenv
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.agents.readonly_context import ReadonlyContext
from pydantic import PrivateAttr

# Load .env for local development; no-op in production (Agent Engine sets env vars)
load_dotenv()

# Module-level OAuth token cache: (access_token, expiry_timestamp)
_oauth_token_cache: Tuple[str, float] = ("", 0.0)
# Refresh the token this many seconds before it actually expires
_TOKEN_REFRESH_BUFFER_SECS = 3600 # 1 hour buffer


def _build_genie_mcp_url() -> str:
    """Construct the Databricks Genie MCP endpoint URL from environment variables."""
    host = os.environ["DATABRICKS_HOST"].rstrip("/")
    space_id = os.environ["GENIE_SPACE_ID"]
    return f"{host}/api/2.0/mcp/genie/{space_id}"


def _fetch_oauth_token() -> Tuple[str, float]:
    """
    Fetch a new Databricks OAuth M2M token using the client credentials grant.

    Required env vars:
        DATABRICKS_HOST        - e.g. https://adb-xxxx.azuredatabricks.net
        DATABRICKS_CLIENT_ID   - Service principal application (client) ID
        DATABRICKS_CLIENT_SECRET - Service principal OAuth secret

    Returns:
        (access_token, expiry_epoch_seconds)
    """
    host = os.environ["DATABRICKS_HOST"].rstrip("/")
    client_id = os.environ["DATABRICKS_CLIENT_ID"]
    client_secret = os.environ["DATABRICKS_CLIENT_SECRET"]

    token_url = f"{host}/oidc/v1/token"
    payload = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "scope": "all-apis",
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode()

    req = urllib.request.Request(
        token_url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(
            f"Failed to obtain Databricks OAuth token from {token_url}: {exc}"
        ) from exc

    access_token = body["access_token"]
    expires_in = int(body.get("expires_in", 3600))
    expiry = time.time() + expires_in
    return access_token, expiry


def _get_oauth_token() -> str:
    """
    Return a valid Databricks OAuth M2M access token, refreshing if expired
    or within _TOKEN_REFRESH_BUFFER_SECS of expiry.
    """
    global _oauth_token_cache
    token, expiry = _oauth_token_cache
    if not token or time.time() >= expiry - _TOKEN_REFRESH_BUFFER_SECS:
        token, expiry = _fetch_oauth_token()
        _oauth_token_cache = (token, expiry)
    return token


def _get_databricks_token() -> str:
    """
    Resolve a Databricks bearer token using the following priority:
      1. DATABRICKS_TOKEN env var (static PAT — returned as-is, no caching needed)
      2. OAuth M2M client credentials (DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET)
      3. Databricks SDK profile fallback (~/.databrickscfg)
    """
    token = os.environ.get("DATABRICKS_TOKEN")
    if token:
        return token

    # OAuth M2M — preferred when running as a service principal (e.g. on Agent Engine)
    if os.environ.get("DATABRICKS_CLIENT_ID") and os.environ.get("DATABRICKS_CLIENT_SECRET"):
        return _get_oauth_token()

    # Fall back to Databricks SDK profile (reads ~/.databrickscfg)
    profile = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")
    try:
        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient(profile=profile)
        return client.config.token
    except Exception as exc:
        raise RuntimeError(
            "Could not resolve a Databricks token. "
            "Set DATABRICKS_TOKEN, or set DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET, "
            "or configure a Databricks profile in ~/.databrickscfg."
        ) from exc


class GenieAgent(BaseAgent):
    """
    Google ADK agent that queries Databricks AI/BI Genie via MCP.

    The inner LlmAgent and MCPToolset are created lazily in set_up() so that
    this class can be safely pickled by Vertex AI Agent Engine without
    serializing open network connections or threading primitives.

    When OAuth M2M credentials are present, McpToolset is configured with a
    header_provider callback instead of a static Authorization header.
    header_provider is invoked by McpToolset on every MCP session creation,
    so _get_oauth_token() (with its module-level expiry cache) ensures a
    fresh token is always used without any manual refresh tracking here.
    """

    model_name: str = "gemini-2.5-flash"

    _inner_agent: Optional[LlmAgent] = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(name="genie_agent", **kwargs)

    # ------------------------------------------------------------------
    # set_up() is called by Vertex AI Agent Engine AFTER deserialization,
    # making it safe to create MCPToolset (which holds live connections).
    # ------------------------------------------------------------------
    def set_up(self) -> None:
        """Initialize the inner LlmAgent with a live MCP connection to Genie."""
        genie_url = _build_genie_mcp_url()

        # When OAuth M2M credentials are available, supply a header_provider so
        # McpToolset fetches a fresh (cached) token on every session creation.
        # For static PATs and SDK-profile tokens, embed the token in the
        # connection headers once — those credentials don't expire.
        use_oauth = bool(
            os.environ.get("DATABRICKS_CLIENT_ID")
            and os.environ.get("DATABRICKS_CLIENT_SECRET")
        )

        if use_oauth:
            def _oauth_header_provider(_: ReadonlyContext) -> dict:
                return {"Authorization": f"Bearer {_get_oauth_token()}"}

            genie_toolset = MCPToolset(
                connection_params=StreamableHTTPConnectionParams(
                    url=genie_url,
                    timeout=30.0,
                    sse_read_timeout=300.0,
                ),
                header_provider=_oauth_header_provider,
            )
        else:
            static_token = _get_databricks_token()
            genie_toolset = MCPToolset(
                connection_params=StreamableHTTPConnectionParams(
                    url=genie_url,
                    headers={"Authorization": f"Bearer {static_token}"},
                    timeout=30.0,
                    sse_read_timeout=300.0,
                ),
            )

        self._inner_agent = LlmAgent(
            model=self.model_name,
            name=self.name,
            description=(
                "A data analyst assistant that answers natural-language questions "
                "about business data stored in Databricks using AI/BI Genie."
            ),
            instruction=(
                "You are a data analyst assistant with access to Databricks AI/BI Genie. "
                "Genie translates your natural-language questions into SQL, executes them "
                "against Unity Catalog tables, and returns structured results.\n\n"
                "Guidelines:\n"
                "- Use the Genie tools to query data when the user asks a data question.\n"
                "- Present query results clearly, using tables or bullet lists as appropriate.\n"
                "- If a query fails or returns no results, explain why and suggest alternatives.\n"
                "- Always cite which data was queried (e.g., 'Based on the sales table...').\n"
                "- Do not fabricate data; only report what Genie returns."
            ),
            tools=[genie_toolset],
        )

    # ------------------------------------------------------------------
    # Core execution: delegate to the inner LlmAgent
    # ------------------------------------------------------------------
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if self._inner_agent is None:
            self.set_up()
        async for event in self._inner_agent.run_async(ctx):
            yield event


# ADK discovers the agent via this module-level variable.
root_agent = GenieAgent()

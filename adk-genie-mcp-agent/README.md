# ADK Genie MCP Agent

Google ADK agent that connects to Databricks AI/BI Genie via MCP (Streamable HTTP) and deploys to GCP Vertex AI Agent Engine.

## Architecture

```
User → Vertex AI Agent Engine
         └── GenieAgent (ADK BaseAgent)
               └── LlmAgent (gemini-2-5-flash)
                     └── MCPToolset → Databricks Genie MCP endpoint
                                        └── AI/BI Genie Space (SQL on Unity Catalog)
```

**Model:** `gemini-2-5-flash` served from Gemini
**MCP endpoint:** `https://<DATABRICKS_HOST>/api/2.0/mcp/genie/<GENIE_SPACE_ID>`

## Project Structure

```
adk-genie-mcp-agent/
├── agent/
│   ├── __init__.py          # exports root_agent (required by ADK)
│   └── agent.py             # GenieAgent definition
├── deployment/
│   ├── deploy.py            # deploy to Vertex AI Agent Engine
│   └── query_remote.py      # query the deployed agent interactively
├── tests/
│   └── test_agent.py
├── run_local.py             # local interactive runner
├── pyproject.toml
├── .env                     # local secrets (not committed)
└── .env.example
```

## Setup

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Configure `.env`

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

Required variables:

| Variable | Description |
|---|---|
| `DATABRICKS_HOST` | Your Databricks workspace URL |
| `GENIE_SPACE_ID` | AI/BI Genie Space ID (from workspace URL) |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | GCP region (e.g. `us-central1`) |
| `STAGING_BUCKET` | GCS bucket for Agent Engine staging |

**Finding your Genie Space ID:** Open your Genie Space in the Databricks workspace. The ID is in the URL:
`https://<host>/genie/spaces/<GENIE_SPACE_ID>`

### Databricks Authentication

Choose **one** of the following authentication methods:

**Option 1 — Personal Access Token (PAT):**
```bash
DATABRICKS_TOKEN=dapi<your-token>
```
Generate at: _User Settings > Developer > Access Tokens_.

**Option 2 — OAuth M2M (Service Principal, recommended for production):**
```bash
DATABRICKS_CLIENT_ID=<service-principal-application-id>
DATABRICKS_CLIENT_SECRET=<oauth-secret>
```
Create a service principal and OAuth secret in the Databricks account console, then grant it access to the workspace and Genie Space. The agent automatically refreshes the token before expiry.

**Option 3 — Databricks SDK profile (`~/.databrickscfg`):**
Set `DATABRICKS_PROFILE=<profile-name>` (defaults to `DEFAULT`) and configure the profile with `databricks configure`.

### 3. Authenticate with GCP

```bash
gcloud auth application-default login
gcloud config set project <YOUR_PROJECT_ID>
```

## Local Development

### Interactive CLI

```bash
python run_local.py
```

### ADK Web UI (recommended)

```bash
adk web
```

Opens a browser-based chat UI at `http://localhost:8000`.

### ADK CLI runner

```bash
adk run agent
```

## Deploy to Vertex AI Agent Engine

```bash
python deployment/deploy.py
```

This will:
1. Wrap the agent with `AdkApp`
2. Upload to GCS staging bucket
3. Create a Vertex AI Agent Engine (Reasoning Engine) endpoint
4. Print the Agent Engine ID on success

Deployment takes ~3-5 minutes.

## Query the Deployed Agent

```bash
python deployment/query_remote.py --engine-id <AGENT_ENGINE_ID>
```

Or set `AGENT_ENGINE_ID` in `.env` and run without arguments.

## Running Tests

```bash
pytest tests/
```

## Key Design Decisions

### `BaseAgent` + `set_up()` pattern

`MCPToolset` holds live network connections and threading primitives that cannot be pickled. Vertex AI Agent Engine serializes agents with `cloudpickle`. To avoid this, `GenieAgent` extends `BaseAgent` and defers `MCPToolset` creation to `set_up()`, which Agent Engine calls **after** deserialization on the container.

### Token resolution

The agent resolves the Databricks bearer token in priority order:

1. `DATABRICKS_TOKEN` env var — static PAT, used as-is
2. `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` — OAuth M2M client credentials grant against `<DATABRICKS_HOST>/oidc/v1/token`. The token is cached module-level and refreshed 1 hour before expiry. `MCPToolset` is configured with a `header_provider` callback so a fresh token is fetched on every MCP session creation.
3. Databricks SDK profile fallback — reads `~/.databrickscfg` using `DATABRICKS_PROFILE` (defaults to `DEFAULT`)

### Model routing

The model string `databricks/databricks-gemini-2-5-flash` uses ADK's LiteLLM integration. The `databricks/` prefix routes the request to the Databricks Foundation Model API OpenAI-compatible endpoint, authenticating with the same Databricks token.

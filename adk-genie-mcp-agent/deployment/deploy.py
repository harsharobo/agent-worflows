"""
Deploy the Genie agent to Vertex AI Agent Engine.

Usage:
    # Install deploy extras first
    pip install -e ".[deploy]"

    # Set required env vars (or load from .env)
    export GOOGLE_CLOUD_PROJECT=your-gcp-project-id
    export GOOGLE_CLOUD_LOCATION=us-central1
    export STAGING_BUCKET=gs://your-staging-bucket
    export DATABRICKS_HOST=https://416411475796958.8.gcp.databricks.com
    export GENIE_SPACE_ID=<your-space-id>

    # Authentication — choose ONE:
    export DATABRICKS_TOKEN=dapi<your-token>                      # Option 1: PAT
    # export DATABRICKS_CLIENT_ID=<client-id>                     # Option 2: OAuth M2M
    # export DATABRICKS_CLIENT_SECRET=<client-secret>

    # Authenticate with GCP
    gcloud auth application-default login

    # Deploy
    python deployment/deploy.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Validate required environment variables before importing heavy SDK modules
_REQUIRED = [
    "DATABRICKS_HOST",
    "GENIE_SPACE_ID",
]
_missing = [v for v in _REQUIRED if not os.environ.get(v)]
if _missing:
    print(f"ERROR: Missing required environment variables: {', '.join(_missing)}")
    print("Set them in your .env file or export them before running this script.")
    sys.exit(1)

import vertexai

from agent.agent import root_agent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]
STAGING_BUCKET = os.environ["STAGING_BUCKET"]
DATABRICKS_HOST = os.environ["DATABRICKS_HOST"].rstrip("/")
GENIE_SPACE_ID = os.environ["GENIE_SPACE_ID"]
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
DATABRICKS_CLIENT_ID = os.environ.get("DATABRICKS_CLIENT_ID", "")
DATABRICKS_CLIENT_SECRET = os.environ.get("DATABRICKS_CLIENT_SECRET", "")
DATABRICKS_API_BASE = os.environ.get("DATABRICKS_API_BASE", "").rstrip("/")

# Validate that at least one auth method is provided
if not DATABRICKS_TOKEN and not (DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET):
    print("ERROR: No Databricks authentication configured.")
    print("  Set DATABRICKS_TOKEN (PAT), or both DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET (OAuth M2M).")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Initialize Vertex AI SDK
# ---------------------------------------------------------------------------
print(f"Initializing Vertex AI SDK...")
print(f"  Project:  {PROJECT_ID}")
print(f"  Location: {LOCATION}")
print(f"  Staging:  {STAGING_BUCKET}")

vertexai.init(project=PROJECT_ID, location=LOCATION)
client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

# ---------------------------------------------------------------------------
# Runtime dependencies installed in the Agent Engine container
# ---------------------------------------------------------------------------
requirements = [
    "google-cloud-aiplatform[agent_engines,adk]>=1.88.0",
    "google-adk>=1.0.0",
    "databricks-sdk>=0.40.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "cloudpickle",
]

# ---------------------------------------------------------------------------
# Environment variables injected into the Agent Engine container
# ---------------------------------------------------------------------------
env_vars = {
    "DATABRICKS_HOST": DATABRICKS_HOST,
    "GENIE_SPACE_ID": GENIE_SPACE_ID,
}
if DATABRICKS_API_BASE:
    env_vars["DATABRICKS_API_BASE"] = DATABRICKS_API_BASE

if DATABRICKS_TOKEN:
    # For development: embed token directly.
    # For production: use Secret Manager reference instead:
    #   env_vars["DATABRICKS_TOKEN"] = {"secret": "databricks-pat", "version": "latest"}
    env_vars["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
    print(
        "\nWARNING: Embedding DATABRICKS_TOKEN directly in Agent Engine config.\n"
        "         For production, use Google Secret Manager:\n"
        '         env_vars["DATABRICKS_TOKEN"] = {"secret": "databricks-pat", "version": "latest"}'
    )
elif DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET:
    # For development: embed credentials directly.
    # For production: use Secret Manager references instead:
    #   env_vars["DATABRICKS_CLIENT_SECRET"] = {"secret": "databricks-client-secret", "version": "latest"}
    env_vars["DATABRICKS_CLIENT_ID"] = DATABRICKS_CLIENT_ID
    env_vars["DATABRICKS_CLIENT_SECRET"] = DATABRICKS_CLIENT_SECRET
    print(
        "\nWARNING: Embedding DATABRICKS_CLIENT_SECRET directly in Agent Engine config.\n"
        "         For production, use Google Secret Manager:\n"
        '         env_vars["DATABRICKS_CLIENT_SECRET"] = {"secret": "databricks-client-secret", "version": "latest"}'
    )

# ---------------------------------------------------------------------------
# Deploy to Vertex AI Agent Engine
# ---------------------------------------------------------------------------
print(f"\nDeploying 'Databricks Genie ADK Agent' to Vertex AI Agent Engine...")
remote_agent = client.agent_engines.create(
    agent=root_agent,
    config={
        "requirements": requirements,
        "agent_framework": "google-adk",
        "display_name": "Databricks Genie ADK Agent",
        "description": "Google ADK agent that queries Databricks AI/BI Genie via MCP.",
        "env_vars": env_vars,
        "staging_bucket": STAGING_BUCKET,
        "extra_packages": ["./agent"],
    }
)

print("\n=== Deployment Successful ===")
print(f"Resource name : {remote_agent}")

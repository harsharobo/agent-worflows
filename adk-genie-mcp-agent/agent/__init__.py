# ADK requires `root_agent` to be importable from the package root.
# Run locally with: adk web agent
from .agent import GenieAgent, root_agent

__all__ = ["root_agent", "GenieAgent"]

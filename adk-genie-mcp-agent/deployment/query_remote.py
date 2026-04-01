"""
Query a deployed Vertex AI Agent Engine instance interactively.

Usage:
    python deployment/query_remote.py --engine-id <AGENT_ENGINE_ID>

    # Or set the engine ID in .env:
    AGENT_ENGINE_ID=1234567890
    python deployment/query_remote.py
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the deployed Genie agent")
    parser.add_argument(
        "--engine-id",
        default=os.environ.get("AGENT_ENGINE_ID"),
        help="Vertex AI Agent Engine ID (reasoning engine ID)",
    )
    parser.add_argument(
        "--user-id",
        default="local-test-user",
        help="User ID for the session",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="GCP project ID",
    )
    parser.add_argument(
        "--location",
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        help="GCP location",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if not args.engine_id:
        print("ERROR: Provide --engine-id or set AGENT_ENGINE_ID in .env")
        raise SystemExit(1)
    if not args.project:
        print("ERROR: Provide --project or set GOOGLE_CLOUD_PROJECT in .env")
        raise SystemExit(1)

    import vertexai
    from google.adk.sessions import VertexAiSessionService
    from google.adk import Runner
    from google.genai import types

    from agent.agent import root_agent

    vertexai.init(project=args.project, location=args.location)

    session_service = VertexAiSessionService(
        project=args.project,
        location=args.location,
        agent_engine_id=args.engine_id,
    )

    runner = Runner(
        agent=root_agent,
        app_name=root_agent.name,
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name=root_agent.name,
        user_id=args.user_id,
    )
    print(f"Session created: {session.id}")
    print("Type your question (or 'quit' to exit):\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input or user_input.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        message = types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )

        print("Agent: ", end="", flush=True)
        async for event in runner.run_async(
            session_id=session.id,
            user_id=args.user_id,
            new_message=message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        print(part.text)
        print()


if __name__ == "__main__":
    asyncio.run(main())

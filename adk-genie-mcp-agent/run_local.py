"""
Run the Genie agent locally for development and testing.

Usage:
    # ADK web UI (recommended):
    adk web agent

    # ADK CLI runner:
    adk run agent

    # Interactive Python CLI:
    python run_local.py
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from agent.agent import root_agent

    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=root_agent.name,
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name=root_agent.name,
        user_id="local-dev-user",
    )

    print("=== Databricks Genie ADK Agent (Local) ===")
    print(f"Genie Space: {os.environ.get('GENIE_SPACE_ID', '<not set>')}")
    print(f"Databricks Host: {os.environ.get('DATABRICKS_HOST', '<not set>')}")
    print(f"Model: databricks-gemini-2-5-flash")
    print("\nType your question (or 'quit' to exit):\n")

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
            user_id="local-dev-user",
            new_message=message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        print(part.text)
        print()


if __name__ == "__main__":
    asyncio.run(main())

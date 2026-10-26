"""Entry point. Ties everything together.

Flow: read the log -> redact secrets -> start the MCP server -> the agent
calls the tools and the LLM writes a structured diagnosis -> print it.

    python main.py sample_log.txt
    python main.py sample_log.txt --verbose          show the tool calls
    python main.py sample_log.txt --retrieval-only   no LLM, no API key needed
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from prepare import extract_error, load_knowledge, redact
from rag import VectorStore, get_embedder

HERE = Path(__file__).parent


def retrieval_only(log_text: str) -> None:
    """Show the first half of the pipeline without calling an LLM."""
    error = extract_error(log_text)
    if not error:
        print("No error found in the log.")
        return
    print(f"Step:    {error['step']}")
    print(f"Error:   {error['message']}\n")
    store = VectorStore(load_knowledge(HERE / "knowledge.json"), get_embedder())
    print("Most similar past failures:")
    for match in store.search(error["message"], k=3):
        print(f"  {match['score']:.3f}  {match['title']}")


async def analyze(log_text: str, verbose: bool) -> None:
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "The optional dependency 'langchain-mcp-adapters' is required to run the CI debugging agent. "
            "Install it with: pip install langchain-mcp-adapters"
        ) from exc

    from llm import build_agent

    client = MultiServerMCPClient(
        {
            "ci-debug": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(HERE / "mcp_server.py")],
                "env": {k: v for k, v in os.environ.items() if k.startswith(("OPENAI", "EMBEDDING"))},
            }
        }
    )
    agent = build_agent(await client.get_tools())
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": f"Analyze this CI log.\n\n{log_text}"}]}
    )

    if verbose:
        for message in result["messages"]:
            for call in getattr(message, "tool_calls", None) or []:
                print(f"[tool call] {call['name']}({str(call['args'])[:80]}...)")
            if message.type == "tool":
                print(f"[tool result] {str(message.content)[:200]}")
        print()

    diagnosis = result["structured_response"]
    if diagnosis.confidence == "low":
        diagnosis.needs_human_review = True

    print(f"Summary:    {diagnosis.summary}")
    print(f"Root cause: {diagnosis.root_cause}")
    print("Fix steps:")
    for number, step in enumerate(diagnosis.fix_steps, 1):
        print(f"  {number}. {step}")
    print(f"Confidence: {diagnosis.confidence}")
    print(f"Needs human review: {diagnosis.needs_human_review}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a failed CI log.")
    parser.add_argument("log", help="path to a CI log file")
    parser.add_argument("--verbose", action="store_true", help="show tool calls")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM")
    args = parser.parse_args()

    load_dotenv(HERE / ".env")
    log_text = redact(Path(args.log).read_text())

    if args.retrieval_only:
        retrieval_only(log_text)
    elif not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Add it to .env, or use --retrieval-only.")
    else:
        asyncio.run(analyze(log_text, args.verbose))


if __name__ == "__main__":
    main()

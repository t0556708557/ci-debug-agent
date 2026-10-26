"""MCP server: exposes our two functions as tools any MCP client can use.

MCP (Model Context Protocol) is a standard way to offer tools to an AI
application. This file runs as its own process. The agent in main.py
starts it, asks which tools exist, and lets the LLM call them.
Do not print to stdout here: stdout carries the MCP messages.
"""
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from prepare import extract_error as find_error
from prepare import load_knowledge
from rag import VectorStore, get_embedder

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

mcp = FastMCP("ci-debug-tools")
store = VectorStore(load_knowledge(HERE / "knowledge.json"), get_embedder())


@mcp.tool()
def extract_error(log_text: str) -> dict:
    """Find the first real error in a CI log. Returns the step, the error message and nearby lines."""
    return find_error(log_text) or {"message": "No error found in the log."}


@mcp.tool()
def search_similar_failures(error_message: str) -> list[dict]:
    """Search past failures similar to this error. Returns matches with a similarity score, the root cause and the fix that worked."""
    return store.search(error_message, k=3)


if __name__ == "__main__":
    mcp.run(transport="stdio")

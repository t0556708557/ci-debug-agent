# CI Debug Agent

A small project that diagnoses a failed GitHub Actions log. It finds the error, searches a small knowledge base of past failures, and asks an LLM for a structured diagnosis with fix steps.

It is built to be read. Each file demonstrates one idea.

## Files

| File | Concept | What it does |
|---|---|---|
| `prepare.py` | Content preparation | Removes secrets from the log and extracts the real error with a few surrounding lines |
| `rag.py` | RAG | Embeddings and a tiny vector store: finds past failures similar to the current error |
| `llm.py` | LLM and structured output | The model, the instructions, and the shape of the answer (`Diagnosis`) |
| `mcp_server.py` | MCP | Offers two functions as tools over the Model Context Protocol |
| `main.py` | The agent | Starts the MCP server, lets the LLM call the tools, prints the result |
| `knowledge.json` | Knowledge base | Five past failures (synthetic examples) |
| `sample_log.txt` | Demo input | A failed Docker push to ECR (synthetic, with a fake key to show redaction) |

## How it works

```
log file
  -> redact secrets                          (prepare.py)
  -> agent starts the MCP server             (main.py, mcp_server.py)
  -> LLM calls extract_error                 (prepare.py)
  -> LLM calls search_similar_failures       (rag.py: embed the error, compare with the stored vectors)
  -> LLM writes a Diagnosis                  (llm.py: structured output)
  -> print it
```

The five ideas, in plain words:

- **Embedding**: text turned into numbers, so similar meanings get similar numbers.
- **Vector store**: a list of those number lists. Searching means finding the closest ones (cosine similarity).
- **RAG**: first retrieve relevant facts, then give them to the LLM, so it answers from your data and not only from memory.
- **Agent**: an LLM that decides which tool to call, and when it has enough information to answer.
- **MCP**: a standard way to publish tools, so any AI application can use them. The server is a separate process from the agent.

## Setup

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then put your OpenAI key in .env
```

## Run

```
python main.py sample_log.txt                    # full run
python main.py sample_log.txt --verbose          # also show the tool calls
python main.py sample_log.txt --retrieval-only   # no LLM, no key needed
```

`--retrieval-only` uses a simple word-counting embedder instead of OpenAI embeddings, so it matches words and not meaning. It is only for seeing the retrieval step work offline.

## Things to try

1. Add a sixth failure to `knowledge.json` and search for it.
2. Change `SYSTEM_PROMPT` in `llm.py` and see how the diagnosis changes.
3. Run with `--verbose` and read the order of the tool calls.
4. Add a second sample log with a different error.
5. Add a third tool to `mcp_server.py` and see the agent discover it.

## Limits

The knowledge base is synthetic and small, there is one model provider, and there is no formal evaluation. The log text passes through the LLM when it calls `extract_error`, which is fine for short logs but not for very large ones.

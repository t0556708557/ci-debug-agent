"""Content preparation: clean a CI log and load the knowledge base.

Two jobs live here:
1. redact() removes secrets, so nothing sensitive is ever sent to the LLM.
2. extract_error() finds the line that actually explains the failure,
   plus a few lines around it, instead of sending the whole log.
"""
import json
import re
from pathlib import Path

SECRET_PATTERNS = [
    r"AKIA[0-9A-Z]{16}",
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"(?i)bearer\s+[A-Za-z0-9._\-]+",
    r"(?i)(password|secret|token|api[_-]?key)\s*[=:]\s*\S+",
]

ERROR_HINT = re.compile(
    r"denied|error:|exception|failed|not found|cannot|unauthorized|forbidden", re.I
)
TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z\s*")


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text


def extract_error(log_text: str, context_lines: int = 5) -> dict | None:
    """Return the first meaningful error, its step name and nearby lines."""
    lines = [TIMESTAMP.sub("", line) for line in log_text.splitlines()]
    for i, line in enumerate(lines):
        if "Process completed with exit code" in line:
            continue
        if ERROR_HINT.search(line):
            step = next(
                (l.split("##[group]")[1] for l in reversed(lines[:i]) if "##[group]" in l),
                "unknown step",
            )
            start = max(0, i - context_lines)
            return {
                "step": step.strip(),
                "message": line.replace("##[error]", "").strip(),
                "context": "\n".join(lines[start : i + context_lines + 1]),
            }
    return None


def load_knowledge(path: str | Path) -> list[dict]:
    """Load past failures and add the text that will be embedded."""
    items = json.loads(Path(path).read_text())
    for item in items:
        item["text"] = f"{item['title']}. {item['error_signature']}. {item['root_cause']}"
    return items

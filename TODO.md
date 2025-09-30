---
this_file: TODO.md
---

# UUTEL Real Provider TODO

- [x] Implement Claude Code provider via CLI subprocess including session management and tool filtering.
- [x] Add cancellation/timeouts for Claude subprocess runner and translate CLI events to chunks.
- [x] Update CLI commands to surface provider-specific requirements and helpful diagnostics.
- [x] Refresh examples to use real providers (with recorded outputs) and add docs for enabling live runs.
- [x] Expand pytest suite with fixture-based tests for each provider plus opt-in live integration tests.
- [x] Capture deterministic CLI fixtures to keep messaging assertions stable offline.
- [] Replace CodexCustomLLM mock output with delegation to CodexUU real provider logic.
- [] Add CLI engine aliases (codex/claude/gemini/cloud) and register UU providers with LiteLLM.
- [] Refresh examples/docs to showcase realistic responses without mock phrasing.

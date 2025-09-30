---
this_file: TODO.md
---

# UUTEL Real Provider TODO

- [ ] Capture sample outputs from Claude, Gemini, Cloud Code, and Codex CLIs/APIs for fixture creation.
- [ ] Document installation and credential prerequisites for each provider in README notes (draft in WORK.md first).
- [ ] Add placeholder pytest cases that mark provider integrations as expected failures until implemented.
- [ ] Implement shared subprocess runner with streaming support in `uutel.core` and cover with unit tests.
- [ ] Extend authentication helpers to read CLI credential files and OAuth tokens with refresh support.
- [ ] Enhance streaming utilities to convert provider-specific chunks into `GenericStreamingChunk` instances.
- [ ] Replace Codex provider mock completion with real HTTP calls including retry + token refresh.
- [ ] Implement Codex streaming path translating SSE chunks to LiteLLM chunks.
- [ ] Add Codex tool/function call translation and associated tests.
- [ ] Build Gemini provider using Google creds (OAuth/API key) with completion, JSON schema, and tool support.
- [ ] Implement Gemini streaming and multimodal (base64 images) handling.
- [ ] Add credential expiry handling + refresh tests for Gemini provider.
- [ ] Implement Cloud Code provider calling `/v1internal:generateContent` with tool + schema support.
- [ ] Add Cloud Code streaming (or long-poll) adapter and usage metadata mapping.
- [ ] Implement Claude Code provider via CLI subprocess including session management and tool filtering.
- [ ] Add cancellation/timeouts for Claude subprocess runner and translate CLI events to chunks.
- [ ] Update CLI commands to surface provider-specific requirements and helpful diagnostics.
- [ ] Refresh examples to use real providers (with recorded outputs) and add docs for enabling live runs.
- [ ] Expand pytest suite with fixture-based tests for each provider plus opt-in live integration tests.
- [ ] Run `uvx hatch test` and record results, coverage, and remaining gaps in WORK + CHANGELOG.

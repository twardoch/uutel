---
this_file: PLAN.md
---

# UUTEL Real Provider Implementation Plan

## Scope
Expose real LiteLLM-compatible providers for Claude Code CLI, Gemini CLI Core, Google Cloud Code, and OpenAI Codex so that `uutel` returns genuine model outputs with streaming, tool calling, and error handling.

## Current State Assessment
- Providers in `src/uutel/providers/*` still return mock completions and streaming tokens.
- Authentication helpers read configs but never call real CLIs or REST APIs.
- CLI commands (`uutel complete`, `uutel list_engines`, etc.) work but only exercise fake provider logic.
- Tests validate formatting and CLI UX but do not exercise live provider flows.
- Dependencies do not yet include vendor SDKs (`anthropic`, `google-generativeai`, Google auth libs, etc.).

## External Provider Characteristics (from repo TLDR references)
- **Claude Code CLI**: Requires local `@anthropic-ai/claude-code` install and `claude login`. Supports multi-turn conversations, configurable allowed tools, streaming stdout events (json lines). Parameters like `temperature` limited; no native structured outputs.
- **Gemini CLI Core**: Wraps Gemini HTTP APIs via OAuth or API key credentials stored under `~/.gemini`. Supports multimodal prompts, JSON schema outputs, tool/function calling. Provides streaming chunks with metadata.
- **Google Cloud Code**: Hits internal Google endpoint `/v1internal:generateContent` requiring OAuth2 (service account or user flow). Requires project ID, supports tool calling and JSON schema injection via prompt transform.
- **OpenAI Codex CLI**: Stores auth at `~/.codex/auth.json`, needs token refresh flow against `https://auth.openai.com/oauth/token`. Uses ChatGPT backend endpoints with specific headers (`chatgpt-account-id`). Supports streaming and tool calls similar to Chat Completions.

## Phase Breakdown

### Phase 2 – Implement OpenAI Codex Provider First (baseline)
- Replace `CodexUU` mock completion with real HTTP calls to ChatGPT backend endpoints using tokens from `CodexAuth` logic.
- Support endpoints for standard completion and streamed SSE (map to `GenericStreamingChunk`).
- Implement tool call conversion to/from OpenAI function call format.
- Add retry logic using existing resilience utilities with detection for 401 (trigger refresh) and rate limit handling.
- Tests: mocked HTTP responses verifying request payloads, token refresh triggered on 401, streaming chunk assembly. Integration test hitting live API guarded behind env flag.
- CLI smoke tests: `uutel complete --engine uutel/codex/...` should return non-mock output when credentials present.

### Phase 3 – Implement Gemini CLI Provider
- ✅ Completed: API-key path using `google-generativeai` with JSON schema tooling, tool call support, and text/multimodal content conversion.
- Implement CLI/OAuth parity for advanced features (multimodal commands, diagnostics) and capture additional recorded fixtures.
- Provide parameter validation (reject unsupported frequency/presence penalties, etc.) with warnings.
- Tests: extend mocked coverage for CLI fallback streaming and refresh edge cases; maintain credential loader refresh assertions.

### Phase 4 – Implement Google Cloud Code Provider
- ✅ Baseline `/v1internal` completion + streaming paths with tool/JSON schema support landed (2025-09-30)
- Implement OAuth2 client using `google-auth-oauthlib` to read Cloud Code credentials (project-specific). Provide fallback for service account JSON path.
- Port message conversion logic from TS: system instructions, tool config, JSON schema injection.
- Call `/v1internal:generateContent` with appropriate headers and handle response structure (candidates, usage metadata, tool calls).
- Support streaming via `:streamGenerateContent` endpoint (if available) or chunk translation from long-poll responses.
- Tests: contract tests with recorded responses, ensure warnings emitted for unsupported settings, verify tool call conversion.

### Phase 5 – Implement Claude Code CLI Provider
- ✅ Completed (2025-10-01): CLI subprocess integration with JSON payload replay, streaming chunk parsing, tool filtering, working directory support, and cancellation hooks.
- ✅ Completed: Fixture-driven unit tests covering completion, streaming, cancellation, and CLI unavailability error messaging.
- Follow-up: consider opt-in live CLI integration tests behind `UUTEL_RUN_LIVE_CLAUDE=1` once credentials available.

### Phase 6 – Documentation, CLI UX, and Validation
- ✅ README updated with live-run instructions and fixture replay notes (2025-10-01).
- Update `DEPENDENCIES.md` explaining new packages and reasoning.
- ✅ CLI help expanded with provider requirements section (2025-10-01).
- ✅ Examples refreshed with recorded provider outputs and replay instructions (2025-10-01).
- ✅ uvx hatch test executed post-changes; log results in WORK.md and CHANGELOG.md.

## Testing & Validation Strategy
- Follow test-first approach: for each provider feature, write failing test capturing expected behaviour before implementation.
- Use pytest fixtures with recorded sample payloads to avoid requiring live network in unit tests.
- Provide opt-in integration tests triggered via env flags (`UUTEL_RUN_LIVE_CODEX=1`, etc.) to validate against real services when credentials present.
- Add functional smoke tests under `examples/` with automation via `./test.sh` script (runs lint + pytest + examples).
- Track coverage improvements; target >=80% overall with focus on new modules.

## Package & Tooling Decisions
- Add `google-auth`, `google-auth-oauthlib`, `google-generativeai` for Google providers.
- Consider `anthropic` only if CLA CLI subprocess proves insufficient; primary plan is CLI subprocess without extra package.
- Use existing `httpx` for Codex HTTP calls; rely on standard library `asyncio` + `subprocess` for CLI streaming.
- Evaluate `aiofiles` only if asynchronous file IO needed for credential caches (otherwise skip).

## Risks & Mitigations
- **Credential availability**: Provide clear error messages and documentation; add `uutel config doctor` enhancements to check auth files.
- **CLI version drift**: Detect CLI version via `--version` and warn if unsupported; store supported version matrix in config.
- **Long-running subprocesses**: Implement timeouts and ensure cleanup; expose config for max duration.
- **API changes**: Wrap HTTP interactions with typed response validation (pydantic models) so tests fail loudly when schemas change.

## Success Criteria
- `uutel complete` returns genuine model output for each provider when creds configured.
- Streaming works end-to-end with chunked responses for all providers.
- Tool/function calling works for providers that support it; unsupported features raise informative errors.
- Test suite covers authentication, request translation, streaming parsing, and error paths; integration tests optional but passing when run with creds.
- Documentation reflects real setup steps; `CHANGELOG.md` records implementation milestones.

## Phase 7 – LiteLLM Adapter Alignment (2025-10-02)

- Replace the remaining mock-only `CodexCustomLLM` shim with a thin delegation layer to `CodexUU` so CLI flows hit real network-backed logic.
- Register all UU providers (`CodexUU`, `ClaudeCodeUU`, `GeminiCLIUU`, `CloudCodeUU`) with LiteLLM and surface canonical engine strings plus provider aliases (`codex`, `claude`, `gemini`, `cloud`) for quick CLI selection.
- Update CLI UX/tests/examples to remove “mock response” phrasing, showcase realistic fixture-backed snippets, and document default model mapping for each alias.
- Ensure new behaviour is covered by unit tests (delegation, alias validation) while remaining offline by patching provider calls.

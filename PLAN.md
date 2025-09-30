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

### Phase 0 – Groundwork & Research Validation
- Verify availability of required CLIs (`claude`, `gemini`, `codex`) and document installation prerequisites in README.
- Capture sample outputs by running each CLI manually; store anonymised fixtures under `tests/data` for replay-based tests.
- Investigate credential file schemas (`~/.claude-code`, `~/.gemini`, `~/.codex`) and plan secure loading; confirm refresh flows.
- Decide on Python packages for API interaction:
  - `google-auth`, `google-auth-oauthlib`, `google-generativeai` for Gemini & Cloud Code.
  - Custom HTTP calls for Codex (no official Python SDK) using `httpx`.
  - `anthropic` optional; likely rely on CLI subprocess for Claude Code.
- Deliverables: research notes in `WORK.md`, fixtures saved, installation steps drafted.
- Tests: add failing placeholder tests that assert fixtures exist and parsers raise `NotImplementedError` until implemented.

### Phase 1 – Core Infrastructure Upgrades
- Create `uutel.core.runners` module with reusable subprocess runner handling stdout streaming, cancellation, and timeouts.
- Implement credential loaders in `uutel.core.auth` for CLI-based providers (Claude, Codex) and OAuth-based providers (Gemini, Cloud Code) with caching and refresh hooks.
- Build generic streaming adapter that converts provider-specific events into `GenericStreamingChunk`.
- Extend error taxonomy in `uutel.core.exceptions` for auth failures, CLI exit codes, HTTP errors.
- Update `uutel.core.utils` to support tool payload encoding/decoding and JSON schema injection helpers.
- Tests: unit tests for subprocess runner (mocked), credential loaders (fixtures), streaming adapter conversions.

### Phase 2 – Implement OpenAI Codex Provider First (baseline)
- Replace `CodexUU` mock completion with real HTTP calls to ChatGPT backend endpoints using tokens from `CodexAuth` logic.
- Support endpoints for standard completion and streamed SSE (map to `GenericStreamingChunk`).
- Implement tool call conversion to/from OpenAI function call format.
- Add retry logic using existing resilience utilities with detection for 401 (trigger refresh) and rate limit handling.
- Tests: mocked HTTP responses verifying request payloads, token refresh triggered on 401, streaming chunk assembly. Integration test hitting live API guarded behind env flag.
- CLI smoke tests: `uutel complete --engine uutel/codex/...` should return non-mock output when credentials present.

### Phase 3 – Implement Gemini CLI Provider
- Build Gemini API client wrapper using `google-generativeai` or direct REST with `google.auth.credentials`. Load OAuth creds from `~/.gemini/oauth_creds.json` or accept API key via config.
- Support text completions, JSON schema responses, tool/function calling (map to LiteLLM tool schema), and image (base64) attachments.
- Implement streaming via SSE or `stream_generate_content` depending on library.
- Provide parameter validation (reject unsupported frequency/presence penalties, etc.) with warnings.
- Tests: mocked responses verifying prompt conversion, schema injection, tool call mapping, streaming. Credential loader test ensures tokens refreshed when expired.

### Phase 4 – Implement Google Cloud Code Provider
- Implement OAuth2 client using `google-auth-oauthlib` to read Cloud Code credentials (project-specific). Provide fallback for service account JSON path.
- Port message conversion logic from TS: system instructions, tool config, JSON schema injection.
- Call `/v1internal:generateContent` with appropriate headers and handle response structure (candidates, usage metadata, tool calls).
- Support streaming via `:streamGenerateContent` endpoint (if available) or chunk translation from long-poll responses.
- Tests: contract tests with recorded responses, ensure warnings emitted for unsupported settings, verify tool call conversion.

### Phase 5 – Implement Claude Code CLI Provider
- Use subprocess runner to invoke `claude api --json --model <model>` (or equivalent) with conversation history piped via stdin.
- Parse incremental JSONL stdout events into streaming chunks; manage CLI session/resume tokens.
- Support configuration for allowed/disallowed tools, file system sandbox path, working directory.
- Implement cancellation by terminating subprocess; handle CLI exit codes/timeouts gracefully.
- Tests: fixture-based tests replaying recorded CLI output to ensure parser yields correct chunks, ensures tool events converted to LiteLLM tool calls. End-to-end test guarded by env var runs real CLI if available.

### Phase 6 – Documentation, CLI UX, and Validation
- Update README with real usage instructions, credential setup steps, troubleshooting.
- Update `DEPENDENCIES.md` explaining new packages and reasoning.
- Expand CLI help to surface provider-specific requirements and environment variables.
- Add examples demonstrating real completions (with recorded outputs) and guidelines for streaming & tool calling.
- Run full test matrix (`uvx hatch test`), measure coverage, document results in `WORK.md` and `CHANGELOG.md`.

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

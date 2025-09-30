---
this_file: WORK.md
---

# 2025-09-30

- /work iteration: targeting Cloud Code provider parity with TS reference (`/v1internal:generateContent` + streaming). Immediate items:
  - Write failing unit tests that exercise Cloud Code completion + streaming using recorded fixtures and tool/JSON schema coverage.
  - Implement Cloud Code request translation, OAuth/API-key auth, and SSE stream parsing with usage metadata mapping.
  - Ensure CLI/docs/TODO updates reflect completed Cloud Code tasks and capture deterministic fixtures for future stability.
  - Added deterministic unit suite `tests/test_cloud_code_provider.py` covering completion auth modes, tool call mapping, and SSE streaming chunks.
  - Replaced `CloudCodeUU` implementation with `/v1internal` request body translation, JSON schema injection, tool declarations, usage propagation, and streaming parser leveraging `create_http_client` utilities.
  - Updated Cloud Code fixture format + TODO/PLAN milestones to reflect delivered functionality.
  - Tests: `uvx hatch test tests/test_cloud_code_provider.py` (4 passed); `uvx hatch test tests/test_provider_fixtures.py` (4 passed) verifying fixture shape.
  - Tests: `uvx hatch test` (225 passed, 2 xfailed placeholders) confirming suite health post-Cloud Code upgrade.

- Reviewed `llms.txt` snapshot alongside provider modules to confirm current implementations still return mock data and rely on direct network calls without retry/credential hygiene. Logged gaps against Vercel provider references (streaming, auth refresh, tooling still missing).
- Restored core utility surface needed by test suite: reintroduced `RetryConfig`, resilient `create_http_client`, tool schema validators/transformers, and response extraction helpers. Added minimal retry wrappers for sync/async `httpx` clients to unblock provider work.
- Exported the recovered utilities via `uutel.core` and package `__init__` so downstream imports match plan/tests.
- Upgraded `CodexUU` to build real HTTP payloads, honour injected clients, and normalise responses; updated test suite to exercise request construction via stubbed clients instead of relying on in-method mocks.
- Tests:
  - `uvx hatch test` → fails (8 CLI assertions) because fixtures still expect legacy `❌ Error: …` phrasing while implementation now emits contextual hints (`❌ … in completion`); remaining provider work also pending.
- Immediate focus: draft provider credential prerequisites in this log before updating README; continue planning streaming/tooling work.
- Updated CLI test expectations to match contextual error helper output and refreshed engine listing assertions.
- Tests:
  - `uvx hatch test` → pass (199 tests) after syncing CLI fixtures with new messaging format.
- /report: `uvx hatch test` (199 passed) confirmed; added `this_file` header to `CHANGELOG.md`, logged test run there, and confirmed no TODO/PLAN items ready for removal.
- /cleanup: executed `make clean` to clear caches (`.pytest_cache`, `.mypy_cache`, `.ruff_cache`, build artifacts).
- Added pytest `xfail` placeholders for Claude, Gemini CLI, and Cloud Code providers to capture expected completion/streaming behaviour without hitting external CLIs/APIs; monkeypatched credential checks to fail fast offline.
- Tests:
  - `uvx hatch test` → pass with 199 tests, 6 xfails (new placeholders)

## Provider Credential Prerequisites (draft for README)

- Codex (ChatGPT backend)
  - Install CLI: `npm install -g @openai/codex@latest` (provides the `codex` binary).
  - Authenticate: `codex login` creates `~/.codex/auth.json` with `access_token` and `account_id`.
  - Fallback: set `OPENAI_API_KEY` to bypass CLI tokens and call standard Chat Completions.
  - Verify: `codex --version` reports the installed release; the CLI auto-refreshes tokens after login.
- Claude Code (Anthropic)
  - Install CLI: `npm install -g @anthropic-ai/claude-code`.
  - Authenticate: `claude login` stores credentials under `~/.claude*/` (CLI manages session refresh).
  - Requirements: Node.js ≥18 with the CLI available on `PATH`.
  - Verify: `claude --version` returns the installed release; rerun `claude login` if the CLI prompts for auth.
- Gemini CLI Core (Google)
  - Install CLI: `npm install -g @google/gemini-cli` (provides the `gemini` binary).
  - Authenticate (option 1): `gemini login` launches OAuth and writes `~/.gemini/oauth_creds.json`.
  - Authenticate (option 2): set one of `GOOGLE_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_GENAI_API_KEY`.
  - Verify: `gemini models list` should succeed once credentials are valid.
- Google Cloud Code AI
  - Shares Gemini credentials: prefers OAuth tokens from `gemini login`, also accepts the same API key env vars.
  - Looks in `~/.gemini/oauth_creds.json`, `~/.config/gemini/oauth_creds.json`, or `~/.google-cloud-code/credentials.json`.
  - Verify: ensure `gemini login` has been run for the target Google account or export `GOOGLE_API_KEY` before invoking Cloud Code models.
  - `uvx hatch test tests/test_utils.py tests/test_tool_calling.py tests/test_codex_provider.py` → pass (71 tests) validating recovered utilities plus the new Codex client workflow.
- /report: reviewed PLAN.md and TODO.md, inspected git status, ran `uvx hatch test` (199 passed, 6 xfailed), pruned completed placeholder task from TODO/PLAN, and logged test run in CHANGELOG.
- /cleanup: executed `make clean` to drop caches (`build/`, `dist/`, `*.egg-info`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, coverage files, `__pycache__`).
- Next: capture recorded CLI fixtures, implement Codex streaming/tool paths, and expand auth refresh coverage for Google providers.
- Iteration focus: capture Gemini/Cloud Code/Codex sample outputs; design shared subprocess runner with streaming; extend authentication helpers for CLI + OAuth flows.
- Progress: Added provider fixture tests and JSON samples; delivered shared subprocess runner with sync/async streaming helpers; expanded auth loader utilities with refresh support; rewrote Codex streaming to parse SSE into `GenericStreamingChunk` and normalised tool call payloads with new tests.
- Tests: `uvx hatch test` (217 passed, 6 xfailed placeholders) plus targeted suites for fixtures, runners, auth, and codex provider.
- /report: reviewed TODO/PLAN, pruned completed entries, ran `uvx hatch test` (217 passed, 6 xfailed placeholders), logged results in CHANGELOG.
- /cleanup: executed `make clean` to clear caches and build artifacts.
- Checked TODO.md — remaining provider tasks pending; proceeding to /work phase to tackle next priority items.
- Iteration TODO: Gemini provider tests, implementation, dependency updates.
- Implemented Gemini provider against google-generativeai with tool/JSON schema support, multimodal message conversion, API/CLI fallback instrumentation.
- Added dedicated Gemini provider tests covering API completions, streaming, CLI fallback refresh, and updated optional dependency for google-generativeai.
- Tests: `uvx hatch test` → 221 passed, 4 xfailed placeholders (CLAUDE/Gemini/Cloud Code pending).

# 2025-10-01

- /report: reviewed PLAN/TODO, ran `uvx hatch test` (225 passed, 2 xfailed placeholders), logged results in CHANGELOG, pruned completed TODO metrics.
- /cleanup: executed `make clean` to clear caches and build artefacts.
- /work iteration goal: clear remaining TODO backlog (Claude provider implementation, CLI diagnostics, docs/examples refresh, deterministic fixtures, expanded tests).
- Immediate tasks:
  - Draft failing tests + recorded fixtures for Claude Code provider completion, streaming, tool filtering, and cancellation/timeouts.
  - Implement Claude Code provider leveraging subprocess runner with cancellation hooks and timeout enforcement.
  - Update CLI diagnostics/help surfaces to highlight provider-specific setup requirements and connectivity checks.
  - Refresh examples with real provider flows (fixture-backed) and document enabling live runs.
  - Expand pytest suite to cover new fixtures, provider behaviours, and CLI diagnostics.
- Authored `tests/test_claude_provider.py` covering completion env propagation, streaming JSONL parsing, cancellation handling, and CLI absence errors; removed placeholder xfails.
- Rebuilt `ClaudeCodeUU` to use subprocess runners, structured env payloads, JSON parsing, streaming chunk assembly, cancellation guards, and async streaming parity.
- Extended CLI `list_engines` output with provider requirement guidance, aligning with updated documentation expectations.
- Refreshed `examples/basic_usage.py` to replay deterministic Claude fixture via monkeypatched provider and printed live-run instructions; executed script to verify output.
- Updated README examples section with offline replay details and live run steps.
- Ran `uvx hatch test` (229 passed, 0 failed) validating full suite after provider + CLI + docs updates.

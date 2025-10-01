---
this_file: PLAN.md
---

# UUTEL Provider Parity Roadmap

## Scope
Deliver production-ready LiteLLM providers for Codex, Gemini CLI, and Google Cloud Code with full parity to their CLI counterparts, realistic fixtures, and hardening focused on reliability rather than new surface area.

## Current Focus
- CLI, config, and fixture guardrails are in place with comprehensive regression coverage (326 tests passing).
- Recorded examples and diagnostics already reflect live provider behaviour.
- Remaining gaps sit in the provider adapters themselves: Codex still lacks live HTTP flows, Gemini CLI parity is partial, and Cloud Code needs OAuth-oriented polish.

## Phase 2 – OpenAI Codex Provider Parity
- Implement real HTTP completion + streaming calls using tokens from `CodexAuth`, handling refresh-token flow and LiteLLM chunk translation.
- Map tool/function call payloads between OpenAI responses and LiteLLM expectations, including streaming deltas.
- Add retry/backoff on 401/429 with targeted unit tests that assert refresh triggers and rate-limit handling.
- Extend CLI smoke tests so `uutel complete --engine codex` uses live outputs when credentials exist, guarded by env flag for CI.

## Phase 3 – Gemini CLI OAuth & Feature Parity
- Bring OAuth-based CLI parity to the Gemini provider, including multimodal prompts and diagnostics output that mirrors the official CLI.
- Validate parameters (temperature, top_p, safety) and emit warnings for unsupported settings with regression coverage.
- Refresh recorded fixtures with realistic Gemini CLI responses covering tool calls and JSON schema output.
- Expand `tests/test_gemini_provider.py` to cover OAuth credential refresh paths and streaming chunk fidelity.

## Phase 4 – Google Cloud Code OAuth & Streaming Polish
- Implement OAuth2 client handling for service-account and user credentials via `google-auth` libraries, ensuring project-aware requests.
- Port Cloud Code message conversion logic (system prompts, JSON schema injection, tool config) from TypeScript references.
- Wire streaming completions through `/v1internal:streamGenerateContent` (or equivalent) translating candidates into LiteLLM chunks.
- Add contract tests with recorded Cloud Code responses plus CLI readiness coverage for OAuth/project edge cases.


## Mini Hardening Sprint – Output Sanitisation & Provider Map Resilience (Planned 2025-10-07)
- **Objective**: Remove lingering reliability footguns in the CLI <> LiteLLM bridge so noisy control sequences and unusual LiteLLM state do not disrupt users invoking UUTEL outside our curated CLI entry points.
- **Tasks**:
  1. Normalise `litellm.custom_provider_map` inputs: teach `setup_providers()` to coerce `None`, tuples, and dict-based formats into a list before merging, while preserving non-UUTEL entries. Add regression cases in `tests/test_cli.py::TestSetupProviders` covering tuple/None/dict inputs to prove we no longer drop third-party handlers.
  2. Introduce terminal-safe output: funnel `_safe_output()` through a control-character scrubber (preserving newlines and standard whitespace) so provider responses containing stray ANSI/VT100 bytes cannot corrupt downstream pipelines. Add focussed tests in `tests/test_cli.py::TestCLIStreamingOutput` (new) that inject ANSI-laden chunks and assert captured stdout/stderr are cleaned.
  3. Case-fold Codex aliases: update `CodexCustomLLM._map_model_name()` to resolve aliases case-insensitively, preventing LiteLLM integrations from failing when partners send uppercase model ids. Extend `tests/test_codex_provider.py::TestCodexCustomLLMModelMapping` with canonical/alias uppercase permutations and a failure assertion for unknown mixed-case inputs.
- **Validation**:
  - Tests-first: add failing pytest cases for each bullet before adjusting implementation.
  - Targeted runs: `uvx hatch test tests/test_cli.py::TestSetupProviders` and the new streaming sanitisation suite, plus `uvx hatch test tests/test_codex_provider.py::TestCodexCustomLLMModelMapping`.
  - Regression sweep: full `uvx hatch test` with results logged to `CHANGELOG.md` and `WORK.md`.



## Testing & Validation Strategy
- Maintain tests-first workflow: introduce failing unit/integration tests for each bullet before implementation.
- Continue using `uvx hatch test` for regression sweeps; document every run in `CHANGELOG.md` and `WORK.md`.
- Refresh fixtures in `tests/data/providers/**` concurrently with provider updates and validate via `tests/test_fixture_integrity.py`.

## Mini Hardening Sprint – Alias & Fixture Alignment
- **Status**: Completed 2025-10-07 (alias alignment + usage guidance hardening).
- **Objective**: Ensure CLI alias usage stays consistent across documentation, fixtures, and runtime helpers so users can invoke providers with shorthand names (`codex`, `claude`, `gemini`, `cloud`) without regressions.
- **Tasks**:
  1. Extend `uutel.core.utils.validate_model_name` to accept the hyphenated canonical engines emitted by the CLI (e.g., `uutel-claude/claude-sonnet-4`) and add regression tests plus example assertions that mark these engines as valid.
  2. Guard recorded fixture metadata by deriving canonical engines via `validate_engine`, and add tests ensuring `engine`/`live_hint` fields stay alias-first so fixtures and docs never drift back to raw provider strings.
  3. Refresh `UUTELCLI.list_engines` usage guidance to highlight alias-first commands for `uutel complete` and `uutel test`, accompanied by CLI snapshot tests that lock in the new help messaging.
- **Validation**:
  - Tests-first: extend `tests/test_examples.py` (fixture metadata) and `tests/test_cli.py` (list output) with failing cases before implementation.
  - Run targeted pytest modules prior to full regression suite for faster feedback.
  - Record updates in `WORK.md`, surface new tests in `CHANGELOG.md` after the regression sweep.

## Mini Hardening Sprint – Codex CustomLLM Reliability (Planned 2025-10-07)
- **Status**: Completed 2025-10-07 (alias coverage, streaming metadata, and model_response normalisation).
- **Objective**: Lock in Codex alias resolution and error propagation so LiteLLM consumers always receive consistent metadata across sync, async, and streaming entry points.
- **Tasks**:
  1. Add regression tests verifying `_map_model_name` resolves both `codex-*` and `my-custom-llm/codex-*` aliases to canonical OpenAI models, including direct passthrough for supported backend IDs.
  2. Cover `streaming` and `astreaming` error paths with tests that assert `litellm.APIConnectionError` instances carry `llm_provider` and canonical `model` metadata when the underlying provider raises `UUTELError`.
  3. Exercise `_prepare_kwargs`'s model_response handling by simulating providers returning `None` or responses lacking `choices/message`, ensuring the helper back-fills a minimal `ModelResponse` structure.
- **Validation**:
  - Write failing tests in `tests/test_codex_provider.py` targeting alias resolution, streaming error metadata, and model response normalisation before touching implementation.
  - Run targeted selections followed by full `uvx hatch test`; log both runs in `WORK.md` and `CHANGELOG.md`.
  - Confirm TODO.md mirrors these tasks and clear entries once regression suite is green.

## Mini Hardening Sprint – Config Normalisation (Planned 2025-10-07)
- **Status**: Completed 2025-10-07
- **Objective**: Ensure persisted configuration values are normalised before validation so CLI runs remain stable even when users edit `.uutel.toml` manually.
- **Tasks**:
  1. Extend `uutel.core.config.load_config` to coerce string/float `max_tokens` values (including signed digits and underscore separators) into integers before validation, and add regression tests in `tests/test_config.py`.
  2. Coerce boolean-like values for `stream`/`verbose` (e.g. "true", "0", "off") to actual booleans to avoid downstream type failures, with accompanying tests covering accepted/rejected literals.
  3. Trim whitespace-only `engine`/`system` strings to `None` so placeholder values do not leak into CLI defaults, and document the behaviour with targeted tests.
- **Validation**:
  - Write failing tests in `tests/test_config.py` capturing each coercion/normalisation scenario before implementation.
  - Run focused pytest selection for config tests, then full `uvx hatch test`; record results in `CHANGELOG.md` and `WORK.md`.


## Mini Hardening Sprint – CLI & Docs Parity (Planned 2025-10-07)
- **Objective**: Lock CLI help output

## Mini Hardening Sprint – Config CLI Input Validation (Planned 2025-10-07)
- **Status**: Completed 2025-10-07
- **Objective**: Harden `uutel config set` input handling so invalid values surface consistent guidance and supported coercions stay regression-tested.
- **Tasks**:
  1. Add regression coverage ensuring `uutel config set` rejects out-of-range or non-numeric `max_tokens`/`temperature` inputs with the current guidance message and without writing new config files.
  2. Verify the `default`/`none` sentinels on boolean flags clear persisted overrides by asserting `uutel config set --stream default --verbose default` removes stored values and updates CLI state.
  3. Accept numeric literals containing underscore separators (e.g. `1_000`) and mixed-case boolean keywords via `uutel config set`, locking behaviour with dedicated tests.
- **Validation**:
  - Introduce failing tests in `tests/test_cli.py::TestCLIConfigCommands` covering each scenario before touching implementation.
  - Patch `save_config` and filesystem interactions in tests to assert no unintended writes occur on validation failure.
  - Run targeted selection (`uvx hatch test tests/test_cli.py::TestCLIConfigCommands`) followed by full suite; record both runs in `WORK.md` and `CHANGELOG.md`.
 and documentation commands to the actual alias/canonical engine mapping so user guidance stays accurate.
- **Tasks**:
  1. Snapshot CLI help output for `uutel`, `uutel complete`, and `uutel test` using `CliRunner`, storing fixtures that assert alias-first examples and parameter guidance stay in sync.
  2. Add documentation lint tests that scan README and troubleshooting guides for `--engine` usage and assert each referenced engine resolves via `validate_engine`, preventing stale aliases.
  3. Guard the README default config snippet by asserting it matches `create_default_config()` output so documentation reflects current defaults.
- **Validation**:
  - Write failing tests covering help snapshots, documentation alias linting, and config snippet parity before implementation.
  - Run targeted pytest selections followed by full `uvx hatch test`; capture outcomes in `WORK.md` and `CHANGELOG.md`.
  - Clear corresponding TODO items once the suite passes and documentation stays synchronized.

## Mini Hardening Sprint – Config CLI Guardrails (Completed 2025-10-07)
- **Objective**: Prevent silent configuration drift by hardening CLI config workflows and locking defaults to the documented snippet.
- **Tasks**:
  1. Detect and reject unknown keyword arguments in `uutel config set` with guidance so typos never silently no-op.
  2. Add a regression test asserting that `uutel config init` writes the exact snippet returned by `create_default_config()` (including trailing newline).
  3. Cover the `uutel config show` no-file branch with a CLI test that verifies the guidance message recommending `uutel config init`.
- **Validation**:
  - Failing pytest cases added in `tests/test_cli.py` prior to implementation ensured the CLI surfaced unknown-key errors and baseline guidance branches.
  - `uvx hatch test` (488 passed, 2 skipped; 16.60s runtime with harness timeout at 21.2s after success) confirmed regressions stayed green and is logged in `CHANGELOG.md`/`WORK.md`.
  - TODO backlog cleared after landing guardrails to keep plan alignment.

## Risks & Mitigations
- Live provider calls must not break CI: guard with environment toggles and rely on replay fixtures by default.
- OAuth flows can leak secrets in logs: scrub sensitive fields and assert masking in tests.
- Streaming translation errors are subtle: use high-fidelity fixtures and helper tests mirroring LiteLLM chunk expectations.

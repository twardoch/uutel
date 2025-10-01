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

## Mini Hardening Sprint – Alias Edge-Case Hardening (Completed 2025-10-07)
- **Objective**: Close remaining alias reliability gaps so CLI shorthands behave predictably even with messy input while keeping error guidance stable for support triage.
- **Tasks**:
  1. Block cross-provider nested shorthand resolution by ensuring `uutel/<alias>/<model>` inputs only resolve when the alias and model belong to the same canonical provider; raise clear guidance otherwise.
  2. Render `validate_engine` error guidance deterministically by alphabetically sorting engine and alias listings, and assert suggestion ordering in tests to avoid future drift.
  3. Trim leading/trailing punctuation (dashes, underscores, stray spaces) in `_normalise_engine_alias` so variants like `--claude--` or `codex-` resolve cleanly, backed by unit and CLI integration tests.
- **Validation**:
  - Added failing cases in `tests/test_cli_validators.py` and `tests/test_cli.py`, confirmed they failed, then reran after implementation to ensure new guardrails passed.
  - Full sweep: `uvx hatch test` -> 578 passed, 2 skipped (26.73s runtime; harness timeout at 33.9s post-success) recorded in `WORK.md`/`CHANGELOG.md`.

## Mini Hardening Sprint – Engine Alias Normalisation (Completed 2025-10-07)
- **Objective**: Reduce support friction by accepting additional human-typed engine shorthands so CLI commands succeed without users memorising exact hyphenation.
- **Outcome**:
  - Introduced `_normalise_engine_alias` to collapse underscores and whitespace into hyphenated segments prior to alias/model lookup.
  - `validate_engine` now resolves `uutel/<model>` shorthands (e.g. `uutel/gpt-4o`) via `MODEL_NAME_LOOKUP` while retaining existing guidance for unknown inputs.
  - Added regression tests covering shorthand resolution for `uutel/gpt-4o`, `gemini_cli`, and `gemini 2.5 pro` to guard future regressions.
- **Validation**:
  - Targeted pytest: `uvx hatch test tests/test_cli_validators.py::TestValidateEngine::test_validate_engine_accepts_model_shorthand_alias tests/test_cli_validators.py::TestValidateEngine::test_validate_engine_normalises_underscore_alias tests/test_cli_validators.py::TestValidateEngine::test_validate_engine_normalises_whitespace_alias`.
  - Full regression: `uvx hatch test` -> 567 passed, 2 skipped (19.17s runtime); captured in `WORK.md` and `CHANGELOG.md`.

## Mini Hardening Sprint – Gemini CLI Payload Guardrails (Completed 2025-10-07)
- **Objective**: Lock Gemini CLI request/response normalisation so malformed CLI payloads and mixed-content prompts are rejected deterministically instead of leaking silent failures.
- **Outcome**:
  - `_completion_via_cli` now invokes `_raise_if_cli_error` prior to JSON extraction, surfacing friendly `UUTELError` messages for CLI faults with preserved code/status context.
  - `_build_contents` folds system prompts into the first user part, skips tool/function call blocks, and `_convert_message_part` handles inline/base64 imagery for API parity.
  - Streaming helpers ignore tool-call events and continue stripping ANSI/OSC bytes so mixed JSON payloads emit clean `GenericStreamingChunk` sequences.
- **Validation**:
  - Targeted pytest selections over the new CLI error, content builder, and streaming sanitisation cases passed after fixes.
  - Full regression: `uvx hatch test` -> 564 passed, 2 skipped (19.91s runtime); results recorded in `WORK.md` and summarised in `CHANGELOG.md`.


## Mini Hardening Sprint – CLI Output Assurance (Completed 2025-10-07)
- **Objective**: Guard CLI-facing surfaces against drift by scrubbing streamed output end-to-end, keeping alias guidance synchronised with data, and auto-syncing usage hints with recorded fixtures.
- **Outcome**:
  - Added CLI integration tests covering control-sequence sanitisation for sync and streaming completions; `_safe_output` already satisfied assertions, locking behaviour into the regression suite.
  - Introduced a docstring alias parity test tying `UUTELCLI.__doc__` summary lines to `ENGINE_ALIASES`, ensuring help snapshots fail fast when mappings change.
  - `list_engines` now renders usage hints from `examples.basic_usage.RECORDED_FIXTURES`, so new recorded transcripts surface automatically without manual CLI updates.
- **Validation**:
  - Targeted pytest selections: `uvx hatch test tests/test_cli.py::TestUUTELCLI::test_complete_command_filters_control_sequences tests/test_cli.py::TestUUTELCLI::test_complete_command_streaming_filters_control_sequences tests/test_cli.py::TestUUTELCLI::test_list_engines_usage_reflects_runtime_recorded_hints tests/test_cli_help.py::test_cli_docstring_alias_guidance_matches_engine_aliases`.
  - Full regression: `uvx hatch test` → 559 passed, 2 skipped (17.89s runtime; harness timeout at 22.8s post-success). Logged in WORK.md/CHANGELOG.md.

## Mini Hardening Sprint – CLI Alias Resilience (Completed 2025-10-07)
- **Objective**: Make alias-driven flows harder to break by accepting nested shorthand inputs, broadcasting all ready-to-run alias hints, and enforcing alias coverage invariants in tests.
- **Outcome**:
  - `validate_engine` now short-circuits canonical engine inputs and resolves nested `uutel/<alias>/<model>` shorthands via a shared `_resolve_candidate` helper.
  - `UUTELCLI.list_engines` renders `uutel test --engine <alias>` lines for every primary alias sourced from `RECORDED_FIXTURES`, keeping CLI guidance alias-first without hard-coded lists.
  - Added alias coverage guard ensuring every canonical engine has either a CLI alias or a model shorthand, plus CLI tests locking gemini/cloud test hints.
- **Validation**:
  - Failing tests captured in `tests/test_cli_validators.py::TestValidateEngine::test_validate_engine_accepts_nested_uutel_model_shorthand` and `tests/test_cli.py::TestUUTELCLI::test_list_engines_command` prior to implementation.
  - Targeted selections (`tests/test_cli_validators.py::TestValidateEngine::test_validate_engine_accepts_nested_uutel_model_shorthand`, `tests/test_cli.py::TestUUTELCLI::test_list_engines_command`, `tests/test_cli.py::TestCLIDiagnostics`) followed by full `uvx hatch test` -> 571 passed, 2 skipped (17.83s runtime).
  - Logged in WORK.md and CHANGELOG.md; TODO backlog cleared post-validation.

## Mini Hardening Sprint – Gemini CLI Parameter Sanitisation (Completed 2025-10-07)
- **Objective**: Guarantee Gemini CLI adapters produce safe request payloads when LiteLLM passes null or malformed optional parameters.
- **Tasks**:
  1. Harden `_build_generation_config` to coerce or fall back to `_DEFAULT_TEMPERATURE`/`_DEFAULT_MAX_TOKENS` when optional params are `None`, booleans, NaN, or out of range; add regression coverage in `tests/test_gemini_provider.py::TestGenerationConfig`.
  2. Normalise `_build_cli_command` inputs so invalid `temperature`/`max_tokens` values default to safe values and `--stream` appears only for truthy flags; cover via new CLI command builder tests.
  3. Sanitize `_build_cli_prompt` to skip `None` content, collapse whitespace, and ensure no literal 'None' leaks into prompts; add targeted prompt assembly tests.
- **Validation**:
  - Added failing coverage for generation-config defaults, CLI command normalisation, and prompt sanitisation prior to implementation.
  - Targeted runs: `uvx hatch test tests/test_gemini_provider.py::TestGenerationConfigDefaults`, `::TestGeminiCLICommandBuilder`, `::TestGeminiCLIPromptBuilder` (generation-config selection completed with harness timeout after success).
  - Full regression: `uvx hatch test` → 555 passed, 2 skipped (20.07s runtime; harness timeout at 25.3s post-success).
  - Logged updates in WORK.md/CHANGELOG.md and cleared TODO entries.

## Mini Hardening Sprint – Alias Synonyms & Example Realism (Completed 2025-10-07)
- **Objective**: Reduce user friction around engine selection and align recorded examples with realistic provider transcripts so docs/tests deliver trustworthy guidance.
- **Outcome**:
  - Extended `ENGINE_ALIASES` with common shorthands (`claude-code`, `gemini-cli`, `cloud-code`, `codex-large`, `openai-codex`) and updated `validate_engine`/diagnostics so synonyms resolve to canonical engines while diagnostics collapse duplicates into grouped summaries.
  - `uutel config set engine <alias>` now handles the new synonyms, persisting canonical identifiers with regression coverage for Fire inputs.
  - Refreshed `tests/data/providers/*/simple_completion.json` transcripts with curated provider outputs, updated example assertions, and added a guard test to prevent regressions back to placeholder copy.
- **Validation**:
  - New failing coverage added in `tests/test_cli_validators.py`, `tests/test_cli.py::TestCLIConfigCommands`, `tests/test_cli.py::TestCLIDiagnostics`, and `tests/test_examples.py` before implementation.
  - Targeted test selections and full `uvx hatch test` (537 passed, 2 skipped; 18.80s runtime) recorded in `WORK.md`/`CHANGELOG.md`.

## Mini Hardening Sprint – Terminal Output & Alias Guardrails (Completed 2025-10-01)
- **Objective**: Harden CLI output sanitisation and alias bookkeeping to prevent regressions as new providers land.
- **Outcome**:
  - `_scrub_control_sequences` now strips 8-bit CSI/OSC/DCS/APC/PM sequences and prunes residual C1 bytes, keeping tmux/screen control strings out of CLI logs.
  - `_safe_output` accepts bytes-like payloads, decoding via UTF-8 with a latin-1 fallback before scrubbing so binary subprocess output renders cleanly.
  - `_build_model_alias_map` raises on duplicate tail aliases (allowing the intentional `gemini-2.5-pro` overlap), preventing silent overwrites as providers expand.
- **Validation**:
  - Tests-first: `uvx hatch test tests/test_cli_helpers.py` and `uvx hatch test tests/test_cli_validators.py::TestValidateEngine::test_build_model_alias_map_raises_on_duplicate_tail` failed on new cases prior to implementation.
  - Targeted: helper/validator selections now pass after updates.
  - Regression: `uvx hatch test` -> 531 passed, 2 skipped (18.98s runtime).

## Mini Hardening Sprint – CLI Error Output Guardrails (Completed 2025-10-07)
- **Objective**: Lock in CLI sanitisation and error metadata helpers with regression-grade coverage.
- **Outcome**:
  - `_scrub_control_sequences` now strips OSC sequences (ESC] ... BEL / ESC\) before ANSI filtering, preventing stray `]0;title` fragments from leaking into logs.
  - `_safe_output` regression tests enforce control-byte scrubbing and BrokenPipe/OSError resilience across stdout and stderr paths.
  - `_extract_provider_metadata` inspects nested kwargs payloads so CLI error formatting consistently surfaces provider/model identifiers.
- **Validation**:
  - Tests-first: `uvx hatch test tests/test_cli_helpers.py` initially reported 3 failures across OSC stripping and kwargs coverage prior to helper adjustments.
  - Targeted: `uvx hatch test tests/test_cli_helpers.py` -> 6 passed confirming helper guardrails.
  - Regression: `uvx hatch test` -> 517 passed, 2 skipped (16.52s runtime; harness timeout at 21.0s post-success) with results logged in `CHANGELOG.md`/`WORK.md`.

## Mini Hardening Sprint – Codex Alias Input Validation (Completed 2025-10-07)
- **Objective**: Ensure Codex alias resolution fails fast with actionable messaging when inputs are blank, non-string, or slightly mistyped.
- **Outcome**:
  - `_map_model_name` now rejects whitespace-only inputs with `BadRequestError("Model name is required")`, covered by new regression tests.
  - Non-string arguments trigger a deterministic `BadRequestError("Model must be a string")`, preventing AttributeErrors from surfacing through LiteLLM.
  - Unknown models emit close-match suggestions (up to three aliases) to guide corrections, with tests asserting message content and ordering.
- **Validation**:
  - Targeted: `uvx hatch test tests/test_codex_provider.py::TestCodexCustomLLMModelMapping` -> 10 passed.
  - Regression: `uvx hatch test` -> 510 passed, 2 skipped (37.96s runtime; harness exit at 47.6s).

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


## Mini Hardening Sprint – Guidance Consistency (Completed 2025-10-07)
- **Objective**: Keep CLI guidance, recorded examples, and README snippets in sync so users always see alias-first commands with accurate provider prerequisites.
- **Outcome**:
  - Added `tests/test_cli.py::TestUUTELCLI::test_list_engines_provider_requirements_cover_all_entries` to lock the provider guidance block against `PROVIDER_REQUIREMENTS`.
  - Introduced `test_list_engines_usage_includes_recorded_live_hints` to assert every `examples.basic_usage.RECORDED_FIXTURES` live hint appears in CLI usage output.
  - Updated README quick usage commands to the recorded live hints and enforced the alignment with `tests/test_documentation_aliases.py::test_readme_quick_usage_includes_recorded_hints`.
- **Validation**:
  - Targeted pytest selections for the new CLI/documentation tests passed after updating the README; full `uvx hatch test` -> 540 passed, 2 skipped (19.29s runtime).
  - TODO entries cleared once CLI, fixtures, and README were in sync.


## Mini Hardening Sprint – Output Sanitisation & Provider Map Resilience (Completed 2025-10-07)
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

## Mini Hardening Sprint – CLI Output Consistency (Completed 2025-10-07)
- **Objective**: Lock down CLI output hygiene and engine metadata to avoid regressions during provider expansion.
- **Outcome**:
  - `_scrub_control_sequences` now strips OSC, DCS, APC, and PM sequences while preserving tabs/newlines, covering tmux-style payloads.
  - Added `_validate_engine_aliases()` to assert alias targets exist in `AVAILABLE_ENGINES`, preventing silent drift in CLI shortcuts.
  - `uutel list_engines` renders engines and aliases alphabetically for deterministic help output and easier snapshot maintenance.
- **Validation**:
  - Targeted pytest selections for CLI helpers, alias invariants, and list ordering now pass.
  - `uvx hatch test` → 521 passed, 2 skipped (23.34s runtime) confirming regression stability.

## Mini Hardening Sprint – Engine Alias & Output Hygiene (Completed 2025-10-01)
- **Objective**: Reduce user-facing friction for common CLI flows by accepting bare model identifiers, avoiding truncated OSC payload data loss, and tightening helper contracts.
- **Tasks**:
  1. Extend `validate_engine` so bare model identifiers (`gpt-4o`, `claude-sonnet-4`, `gemini-2.5-pro`) resolve to the documented canonical engines while preserving suggestion guidance for unknown inputs.
  2. Reproduce the truncated OSC/DCS case where `_scrub_control_sequences` drops trailing text when the terminator is missing; adjust the scanner to discard only the malformed sequence while keeping subsequent user output.
  3. Harden `_safe_output` by rejecting unknown `target` arguments so future call-site typos surface immediately; update helper tests for stdout/stderr flows and the new failure mode.
- **Validation**:
  - Added failing coverage in `tests/test_cli_validators.py::TestValidateEngine::test_validate_engine_accepts_bare_model_identifiers`, `tests/test_cli_helpers.py::TestScrubControlSequences::test_preserves_text_after_truncated_sequences`, and `tests/test_cli_helpers.py::TestSafeOutput::test_rejects_unknown_target` before implementation.
  - Targeted pytest selections for validators/helpers passed after fixes; full `uvx hatch test` -> 528 passed, 2 skipped (22.20s runtime; harness timeout at 28.0s immediately after success) logged in `WORK.md`/`CHANGELOG.md`.
  - README guidance already referenced canonical engine names; no copy changes required.

## Risks & Mitigations
- Live provider calls must not break CI: guard with environment toggles and rely on replay fixtures by default.
- OAuth flows can leak secrets in logs: scrub sensitive fields and assert masking in tests.
- Streaming translation errors are subtle: use high-fidelity fixtures and helper tests mirroring LiteLLM chunk expectations.

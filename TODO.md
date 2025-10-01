---
this_file: TODO.md
---

# UUTEL Real Provider TODO

- [] Harden `setup_providers()` to normalise non-list `litellm.custom_provider_map` inputs and preserve third-party handlers with new regression tests.
- [] Scrub ANSI/control sequences inside CLI output helpers and add streaming output sanitisation coverage.
- [] Make `CodexCustomLLM` alias resolution case-insensitive and expand model mapping tests for uppercase permutations.

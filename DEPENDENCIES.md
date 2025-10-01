# UUTEL Dependencies

This document lists all dependencies used by UUTEL and the rationale for including each.

## Core Dependencies

### LiteLLM Integration
- **litellm>=1.70.0**: Core requirement for extending LiteLLM's provider ecosystem
  - Provides `CustomLLM` base class that `BaseUU` extends
  - Enables seamless integration with LiteLLM's unified interface
  - Required for model registration and routing

### HTTP and Async Support
- **httpx>=0.25.0**: Modern async-capable HTTP client
  - Used for provider API communications
  - Supports both sync and async operations
  - Better type hints and modern Python support than requests
  - Required for reliable connection management

- **aiohttp>=3.8.0**: Async HTTP server/client framework
  - Used for async HTTP operations where httpx is insufficient
  - Provides WebSocket support for streaming responses
  - Required for advanced async communication patterns

### Data Validation and Parsing
- **pydantic>=2.0.0**: Data validation and settings management
  - Ensures type safety for request/response models
  - Provides runtime validation of provider responses
  - Modern Pydantic v2 for performance and features

- **pydantic-settings>=2.0.0**: Configuration management with Pydantic
  - Handles environment variable loading
  - Provides structured configuration validation
  - Integrates seamlessly with main Pydantic models

- **jsonschema>=4.23.0**: JSON Schema validation utilities
  - Validates recorded provider fixtures to catch response-shape drift
  - Provides informative errors for malformed sample payloads
  - Lightweight runtime impact by limiting usage to the test suite

- **tomli-w>=1.0.0**: Minimal TOML writer used for persisting the CLI config file
  - Ensures quotes and newlines are escaped correctly without bespoke serializers
  - Pure-Python library maintained alongside `tomli`, so no compiled wheels required
  - Pairs with Python 3.11+ `tomllib` reader for reliable round-trip configuration support

- **tomli>=1.2.0**: Backport TOML parser for Python 3.10 support
  - Mirrors Python 3.11's `tomllib` so config loading works on older interpreters
  - Loaded conditionally via environment markers to avoid redundant installs on 3.11+
  - Keeps `load_config` logic portable without bundling a custom parser

### Logging and Monitoring
- **loguru>=0.7.0**: Modern Python logging library
  - Simpler and more powerful than standard logging
  - Better structured logging support
  - Excellent for debugging provider interactions

- **psutil>=5.9.0**: Process and system metrics helper
  - Powers troubleshooting scripts that report memory/CPU usage before live runs
  - Underpins optional smoke tests that assert resource ceilings on CI machines
  - Upgraded to 7.x in dev extras for more detailed platform statistics when profiling

## Authentication Dependencies

### Google Cloud and OAuth
- **google-auth>=2.15.0**: Google authentication library
  - Required for Google Cloud Code and Gemini CLI providers
  - Handles service account and OAuth flows
  - Industry standard for Google API authentication

- **google-auth-oauthlib>=1.0.0**: OAuth 2.0 flows for Google APIs
  - Required for interactive OAuth flows
  - Supports PKCE for secure OAuth in Claude Code provider
  - Handles token refresh and storage

- **google-generativeai>=0.7.0**: Official Gemini Python bindings
  - Provides typed client for Gemini `generate_content` API
  - Handles streaming responses and native tool/JSON schema support
  - Reduces custom HTTP code for Gemini provider implementation

- **google-cloud-core>=2.0.0**: Core Google Cloud client library
  - Provides common utilities for Google Cloud services
  - Required for Cloud Code provider implementation
  - Ensures consistent error handling across Google services

- **google-api-python-client>=2.130.0**: Low-level Google API discovery client
  - Used by Cloud Code extras for project metadata lookups and OAuth token validation
  - Provides a REST fallback when gcloud CLIs are unavailable on developer machines
  - Keeps traffic authenticated through the same credentials loaded by `google-auth`

- **google-cloud-aiplatform>=1.55.0**: Vertex AI SDK for Gemini Enterprise endpoints
  - Enables Cloud Code and Gemini CLI extras to call hosted models with first-party helpers
  - Surfaces long-running operation polling utilities that simplify streaming completions
  - Includes dependency pins that align with current Google API versions

- **google-cloud-resource-manager>=1.12.3**: Project and folder metadata client
  - Lets readiness checks confirm the active Google Cloud project before issuing requests
  - Provides IAM policy inspection so the CLI can surface actionable permission errors
  - Keeps project selection logic consistent with gcloud defaults

### Provider Credential Helpers
- **browser-cookie3>=0.19.1**: Cross-browser cookie export utility
  - Allows the Codex and Claude extras to reuse session cookies from local browsers
  - Simplifies non-API authentication flows by reading existing `*.anthropic.com` and `*.openai.com` cookies
  - Keeps automation lightweight without shipping bespoke keychain integrations

- **keyring>=24.3.1**: System keychain abstraction
  - Stores refreshed session tokens securely when CLI extras perform OAuth/device logins
  - Avoids writing long-lived credentials to plain-text config files
  - Works across macOS Keychain, Windows Credential Store, and common Linux secret backends

- **selenium>=4.18.0**: Browser automation toolkit
  - Provides a headless fallback to complete vendor login flows when CLI utilities require web auth
  - Useful for recording deterministic fixtures against live environments without manual interaction
  - Bundled only with the Claude CLI extra to keep the default install lightweight

- **requests>=2.31.0**: Ubiquitous synchronous HTTP client
  - Powers small readiness probes and metadata fetches inside the Claude/Codex extras where httpx is unnecessary
  - Offers compatibility with vendor helper scripts that expect `requests` objects
  - Remains optional so the core package stays async-first via httpx

### Codex API Fallbacks
- **openai>=1.35.0**: Official OpenAI Python bindings
  - Enables the Codex provider to fall back to direct OpenAI API calls when session cookies are unavailable
  - Provides typed request/response models that align with LiteLLM expectations
  - Includes automatic retry/backoff utilities reused by the CLI diagnostics command

- **tiktoken>=0.7.0**: OpenAI tokenizer bindings
  - Supplies accurate token counting for Codex transcripts when generating usage metadata
  - Keeps CLI diagnostics aligned with server-side billing calculations
  - Loaded lazily so fixtures tests remain fast when token stats are not required

## CLI and User Interface Dependencies

### Command Line Interface
- **fire>=0.7.1**: Reflection-based CLI framework
  - Powers the `uutel` command surface with minimal boilerplate
  - Maps Python objects to subcommands while preserving type hints
  - Lightweight dependency compared to Click/Typer stacks

- **click>=8.1.0**: Optional CLI helper for ecosystem compatibility
  - Enables future integration with Click-based tooling
  - Included in extras for developers extending the CLI surface

- **typer>=0.12.0**: Rich CLI ergonomics built on Click
  - Provides a type-safe CLI builder for future interactive subcommands
  - Used in examples and tooling scripts where auto-generated help and prompts improve UX
  - Bundled in the `cli` extra so the base install stays dependency-light

- **rich>=13.7.0**: Terminal formatting and progress rendering
  - Powers optional live diagnostics output (tables, status spinners, syntax highlighting)
  - Keeps CLI messaging readable when streaming multi-line guidance to developers
  - Only pulled in with the `cli` extra to avoid shipping heavy formatting defaults

## Development and Testing Dependencies

### Testing Framework
- **pytest>=8.3.4**: Modern Python testing framework
  - Comprehensive test suite support
  - Better fixtures and parametrization than unittest
  - Industry standard for Python testing

- **pytest-cov>=6.0.0**: Coverage reporting for pytest
  - Measures test coverage across codebase
  - Ensures quality through coverage metrics
  - Required for comprehensive testing

- **pytest-xdist>=3.6.1**: Distributed testing for pytest
  - Enables parallel test execution
  - Faster test runs for large test suites
  - Improves development workflow

- **pytest-asyncio>=0.25.3**: Async testing support
  - Required for testing async provider methods
  - Handles event loops properly in tests
  - Essential for async/await test cases

- **pytest-mock>=3.15.0**: Mocking framework for pytest
  - Provides mocking capabilities for external APIs
  - Enables isolated unit testing
  - Required for testing without hitting real APIs

- **coverage[toml]>=7.6.12**: Coverage measurement tool
  - Tracks test coverage metrics
  - TOML configuration support
  - Required for coverage reporting

### Code Quality Tools
- **ruff>=0.9.7**: Fast Python linter and formatter
  - Modern replacement for flake8 and black with built-in import sorting
  - Extremely fast linting and formatting
  - Single tool for multiple code quality checks

- **mypy>=1.15.0**: Static type checker
  - Ensures type safety throughout codebase
  - Catches type-related bugs before runtime
  - Required for maintaining type hints quality

- **pre-commit>=4.1.0**: Git pre-commit hooks framework
  - Ensures code quality before commits
  - Automatically runs linting and formatting
  - Prevents broken code from entering repository

- **bandit>=1.8.0**: Security linter for Python code
  - Scans code for common security issues
  - Identifies potential vulnerabilities
  - Required for maintaining security standards

- **safety>=4.0.0**: Dependency vulnerability scanner
  - Checks dependencies for known security vulnerabilities
  - Provides security alerts for outdated packages
  - Required for maintaining secure dependencies

### Import Organization

### Documentation Tooling
- **mkdocs>=1.6.0**: Static documentation site generator
  - Powers the optional developer handbook published from the `docs/` directory
  - Provides built-in live-reload so contributors can preview docs locally
  - Lightweight theme layer that plays nicely with mkdocs-material customization

- **mkdocs-material>=9.5.0**: Feature-rich MkDocs theme
  - Supplies search, tabs, and responsive layout for the documentation site
  - Ships with accessible colour palettes suited to long-form provider docs
  - Bundled in the `docs` extra only, keeping runtime installs slim

- **mkdocstrings[python]>=0.25.0**: Automatic API reference generator
  - Extracts docstrings from provider modules into the published docs
  - Ensures API docs stay synchronized with code changes without manual duplication
  - Python handler selected to keep configuration minimal

- **mkdocs-gen-files>=0.5.0**: Dynamic page generation helper
  - Used to assemble changelog snapshots and configuration guides during doc builds
  - Lets us mirror README sections into the documentation site without copy/paste drift
  - Keeps doc generation logic in Python rather than shell scripts

### Profiling Utilities
- **py-spy>=0.3.0**: Sampling profiler for production processes
  - Helps diagnose slow provider interactions without instrumenting the codebase
  - Generates flamegraphs referenced in the troubleshooting guide
  - Included in the `profile` extra so everyday installs avoid heavy native wheels

- **memory-profiler>=0.61.0**: Line-level memory usage tracker
  - Validates that streaming implementations do not leak large buffers under load
  - Used in regression notebooks to spot regressions before shipping
  - Optional extra to keep default dependencies pure-Python

- **line-profiler>=4.1.0**: Deterministic line timing profiler
  - Complements `py-spy` by pinpointing hot paths inside synchronous helpers
  - Handy when tuning CLI output formatting and JSON normalization steps
  - Enabled through the `profile` extra for targeted performance investigations

### Development Tooling
- **uv>=0.2.0**: Fast Python packaging and environment manager
  - Standardizes local workflows (`uv sync`, `uv run`) across docs, tests, and release scripts
  - Ensures lockfiles remain reproducible without invoking pip directly
  - Powers the release pipeline invoked via `build.sh` and CI automation
## Package Selection Rationale

### Why These Specific Versions?
- **Minimum versions**: Chosen to ensure compatibility with older Python installations
- **Modern libraries**: Selected for better type hints, async support, and performance
- **Stable APIs**: All dependencies have stable, well-documented APIs
- **Active maintenance**: All packages are actively maintained with regular updates

### Alternative Considerations
- **requests vs httpx**: httpx chosen for async support and modern Python compatibility
- **argparse vs fire**: Fire chosen for its lightweight reflection-based command mapping
- **unittest vs pytest**: pytest chosen for better fixtures and more readable tests
- **black vs ruff**: ruff chosen for speed and comprehensive functionality

## Security Considerations
- All dependencies are from trusted sources (PyPI verified packages)
- Version pinning prevents unexpected updates that could introduce vulnerabilities
- Regular dependency updates planned to address security issues
- No dependencies with known security vulnerabilities

## Future Dependency Plans
- Consider adding `tenacity` for more sophisticated retry logic
- May add `click-completion` for shell completions
- Consider `structlog` for more structured logging
- Evaluate `pydantic-extra-types` for specialized validation types

## Extra Bundle Aliases
- **uutel**: Meta-package used inside extras (e.g. `uutel[providers]`, `uutel[full]`)
  - Allows the lockfile to express umbrella installs that pull in multiple optional groups at once
  - Keeps installation commands predictable for contributors (`uv add uutel[full]` during development)
  - Avoids duplicating long dependency lists across extras by reusing the primary distribution

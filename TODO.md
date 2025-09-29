# UUTEL Implementation TODO

## Phase 1: Core Infrastructure (Days 1-2)

### Package Structure Setup
- [ ] Create core package structure (uutel/, core/, providers/)
- [ ] Setup __init__.py files with proper imports
- [ ] Configure pyproject.toml with dependencies and metadata
- [ ] Setup basic package imports and exports

### Base Provider Class
- [ ] Implement BaseUU class in core/base.py
- [ ] Define abstract methods for completion, streaming, async operations
- [ ] Add provider name and model identification
- [ ] Implement LiteLLM BaseLLM inheritance pattern

### Authentication Framework
- [ ] Create BaseAuth abstract class in core/auth.py
- [ ] Implement OAuthAuth class for OAuth 2.0 flows
- [ ] Implement ApiKeyAuth class for API key authentication
- [ ] Implement ServiceAccountAuth class for Google Cloud
- [ ] Add token refresh and validation methods

### Message Transformation Utilities
- [ ] Create transformation functions in core/utils.py
- [ ] Implement transform_openai_to_provider function
- [ ] Implement transform_provider_to_openai function
- [ ] Implement handle_tool_calls function
- [ ] Add streaming chunk transformation utilities

### Exception Handling
- [ ] Define custom exception classes in core/exceptions.py
- [ ] Create provider-specific error types
- [ ] Add authentication error handling
- [ ] Implement error mapping from provider APIs

## Phase 2: Provider Implementations (Days 3-6)

### Claude Code Provider
- [ ] Create providers/claude_code/ directory structure
- [ ] Implement ClaudeCodeUU class in provider.py
- [ ] Port OAuth authentication from AI SDK implementation
- [ ] Implement browser-based auth flow
- [ ] Add MCP tool calling support
- [ ] Create message transformation for Claude API
- [ ] Implement streaming support
- [ ] Add error handling and retry logic

### Gemini CLI Provider
- [ ] Create providers/gemini_cli/ directory structure
- [ ] Implement GeminiCLIUU class in provider.py
- [ ] Port multi-auth support (API key, Vertex AI, OAuth)
- [ ] Integrate with Gemini CLI Core patterns
- [ ] Implement advanced tool calling capabilities
- [ ] Create message/tool transformation
- [ ] Add streaming support
- [ ] Handle different authentication types

### Cloud Code Provider
- [ ] Create providers/cloud_code/ directory structure
- [ ] Implement CloudCodeUU class in provider.py
- [ ] Port Google Cloud Code authentication
- [ ] Implement Code Assist API integration
- [ ] Add OAuth flow with project setup
- [ ] Create message transformation for Cloud Code API
- [ ] Implement service account authentication
- [ ] Add streaming support

### Codex Provider
- [ ] Create providers/codex/ directory structure
- [ ] Implement CodexUU class in provider.py
- [ ] Port session token management from AI SDK
- [ ] Implement token refresh mechanisms
- [ ] Add Codex CLI compatibility
- [ ] Create streaming implementation
- [ ] Handle ChatGPT backend integration
- [ ] Add fallback to OpenAI API key

## Phase 3: LiteLLM Integration (Day 7)

### Provider Registration
- [ ] Update main __init__.py with provider imports
- [ ] Configure litellm.custom_provider_map registration
- [ ] Implement setup_model_routing function
- [ ] Add model prefix routing (uutel/provider/model)
- [ ] Test provider discovery and routing

### Provider Factory Functions
- [ ] Implement create_claude_code_provider factory
- [ ] Implement create_gemini_cli_provider factory
- [ ] Implement create_cloud_code_provider factory
- [ ] Implement create_codex_provider factory
- [ ] Add provider configuration validation
- [ ] Test factory function creation and initialization

### LiteLLM Compatibility
- [ ] Test completion API compatibility
- [ ] Test streaming API compatibility
- [ ] Test tool calling integration
- [ ] Verify error handling matches LiteLLM patterns
- [ ] Test async operation support

## Phase 4: Examples and Documentation (Day 8)

### Basic Usage Examples
- [ ] Create examples/basic_usage.py
- [ ] Add examples for all four providers
- [ ] Test completion calls for each provider
- [ ] Verify model routing works correctly

### Streaming Examples
- [ ] Create examples/streaming_example.py
- [ ] Test async streaming for each provider
- [ ] Add error handling examples
- [ ] Verify streaming chunk handling

### Tool Calling Examples
- [ ] Create examples/tool_calling_example.py
- [ ] Test function calling with each provider
- [ ] Add complex tool scenarios
- [ ] Verify tool result handling

### Documentation
- [ ] Update README.md with installation instructions
- [ ] Add provider-specific configuration docs
- [ ] Create API documentation
- [ ] Add troubleshooting guide

## Phase 5: Testing and Validation (Day 9)

### Core Tests
- [ ] Create conftest.py with test fixtures
- [ ] Implement test_base.py for BaseUU functionality
- [ ] Implement test_auth.py for authentication classes
- [ ] Implement test_utils.py for transformation utilities
- [ ] Implement test_exceptions.py for error handling
- [ ] Implement test_providers_init.py for provider initialization

### Provider-Specific Tests
- [ ] Create test_claude_code.py with mocked API tests
- [ ] Create test_gemini_cli.py with mocked API tests
- [ ] Create test_cloud_code.py with mocked API tests
- [ ] Create test_codex.py with mocked API tests
- [ ] Test authentication flows (mocked)
- [ ] Test message transformation
- [ ] Test tool calling functionality

### Integration Tests
- [ ] Create test_integration.py for LiteLLM integration
- [ ] Test provider registration with LiteLLM
- [ ] Test model routing and discovery
- [ ] Test error propagation
- [ ] Test streaming integration

### Test Coverage and Quality
- [ ] Run pytest with coverage reporting
- [ ] Ensure >90% test coverage
- [ ] Run mypy for type checking
- [ ] Verify all tests pass
- [ ] Test examples run successfully

## Phase 6: Package Distribution (Day 10)

### Package Configuration
- [ ] Finalize pyproject.toml with all metadata
- [ ] Configure optional dependencies for each provider
- [ ] Add entry points if needed
- [ ] Test package building with `uv build`

### Quality Assurance
- [ ] Run code formatting with ruff
- [ ] Run linting with ruff
- [ ] Fix any type checking issues
- [ ] Ensure all examples work
- [ ] Test installation from built package

### Documentation Finalization
- [ ] Update DEPENDENCIES.md with dependency rationale
- [ ] Create migration guide from existing implementations
- [ ] Add API reference documentation
- [ ] Update CHANGELOG.md with initial release

### Release Preparation
- [ ] Tag version 1.0.0
- [ ] Test package installation in clean environment
- [ ] Verify all providers work with LiteLLM
- [ ] Create release notes

## Implementation Guidelines

### Code Quality Checklist
- [ ] All functions under 20 lines
- [ ] All files under 200 lines
- [ ] No enterprise patterns or abstractions
- [ ] Clear, readable code with minimal complexity
- [ ] Proper type hints throughout
- [ ] Comprehensive docstrings

### Testing Requirements
- [ ] Unit tests for all public methods
- [ ] Integration tests for LiteLLM compatibility
- [ ] Mocked authentication tests
- [ ] Message transformation tests
- [ ] Streaming functionality tests
- [ ] Tool calling tests
- [ ] Error handling tests

### Dependencies Validation
- [ ] Minimal core dependencies (litellm, httpx, pydantic)
- [ ] Optional provider-specific dependencies
- [ ] No unnecessary enterprise libraries
- [ ] Compatible versions across dependencies

### Performance Verification
- [ ] Provider initialization under 100ms
- [ ] Request transformation under 10ms
- [ ] No memory leaks in streaming scenarios
- [ ] Graceful network timeout handling

## Success Criteria
- [ ] All four providers registered with LiteLLM
- [ ] Completion and streaming work for each provider
- [ ] Tool calling functions correctly where supported
- [ ] Authentication flows work for all auth types
- [ ] Error handling provides clear messages
- [ ] Message transformation maintains fidelity
- [ ] Package installable via pip/uv
- [ ] Examples run without errors
- [ ] Tests pass with >90% coverage
- [ ] Compatible with LiteLLM Router and proxy
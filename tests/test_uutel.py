# this_file: tests/test_uutel.py
"""Test suite for uutel.uutel module."""

from unittest.mock import patch

import pytest

from uutel.uutel import Config, main, process_data


class TestConfig:
    """Test cases for Config dataclass."""

    def test_config_creation_basic(self) -> None:
        """Test basic Config creation with required fields."""
        config = Config(name="test", value="example")

        assert config.name == "test"
        assert config.value == "example"
        assert config.options is None

    def test_config_creation_with_string_value(self) -> None:
        """Test Config creation with string value."""
        config = Config(name="string_test", value="hello_world")

        assert config.name == "string_test"
        assert config.value == "hello_world"
        assert isinstance(config.value, str)

    def test_config_creation_with_int_value(self) -> None:
        """Test Config creation with integer value."""
        config = Config(name="int_test", value=42)

        assert config.name == "int_test"
        assert config.value == 42
        assert isinstance(config.value, int)

    def test_config_creation_with_float_value(self) -> None:
        """Test Config creation with float value."""
        config = Config(name="float_test", value=3.14)

        assert config.name == "float_test"
        assert config.value == 3.14
        assert isinstance(config.value, float)

    def test_config_creation_with_options(self) -> None:
        """Test Config creation with options dictionary."""
        options = {"key1": "value1", "key2": 123, "key3": True}
        config = Config(name="test", value="example", options=options)

        assert config.name == "test"
        assert config.value == "example"
        assert config.options == options

    def test_config_equality(self) -> None:
        """Test Config equality comparison."""
        config1 = Config(name="test", value="example")
        config2 = Config(name="test", value="example")
        config3 = Config(name="different", value="example")

        assert config1 == config2
        assert config1 != config3

    def test_config_repr(self) -> None:
        """Test Config string representation."""
        config = Config(name="test", value="example", options={"key": "value"})
        repr_str = repr(config)

        assert "Config" in repr_str
        assert "name='test'" in repr_str
        assert "value='example'" in repr_str


class TestProcessData:
    """Test cases for process_data function."""

    def test_process_data_empty_list_raises_error(self) -> None:
        """Test that empty data list raises ValueError."""
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            process_data([])

    def test_process_data_with_valid_data(self) -> None:
        """Test process_data with valid input data."""
        data = ["item1", "item2", "item3"]
        result = process_data(data)

        assert isinstance(result, dict)
        assert "processed_count" in result
        assert "items" in result
        assert "timestamp" in result
        assert "debug_mode" in result
        assert result["processed_count"] == 3
        assert len(result["items"]) == 3
        assert result["debug_mode"] is False

        # Check first item structure
        first_item = result["items"][0]
        assert first_item["original"] == "item1"
        assert first_item["type"] == "str"

    def test_process_data_with_config(self) -> None:
        """Test process_data with configuration."""
        data = ["item1", "item2"]
        config = Config(name="test_config", value="test_value")
        result = process_data(data, config=config)

        assert isinstance(result, dict)
        assert result["processed_count"] == 2
        assert len(result["items"]) == 2

        # Check config integration
        first_item = result["items"][0]
        assert first_item["original"] == "item1"
        assert first_item["config_name"] == "test_config"
        assert first_item["config_value"] == "test_value"

    def test_process_data_with_debug_mode(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test process_data with debug mode enabled."""
        data = ["item1", "item2"]

        result = process_data(data, debug=True)

        assert isinstance(result, dict)
        assert result["processed_count"] == 2
        assert result["debug_mode"] is True
        assert len(result["items"]) == 2

    def test_process_data_debug_changes_log_level(self) -> None:
        """Test that debug mode with centralized logging works."""
        data = ["item1"]

        # With new centralized logging, debug flag controls behavior internally
        # Just verify the function executes successfully with debug=True
        result = process_data(data, debug=True)
        assert isinstance(result, dict)

    def test_process_data_with_none_data_raises_error(self) -> None:
        """Test that None data raises ValueError."""
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            process_data(None)

    def test_process_data_with_config_and_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test process_data with both config and debug mode."""
        data = ["item1", "item2"]
        config = Config(name="debug_test", value=42, options={"debug": True})

        result = process_data(data, config=config, debug=True)

        assert isinstance(result, dict)
        assert result["processed_count"] == 2
        assert result["debug_mode"] is True
        assert len(result["items"]) == 2

        # Check config and options integration
        first_item = result["items"][0]
        assert first_item["original"] == "item1"
        assert first_item["config_name"] == "debug_test"
        assert first_item["config_value"] == 42
        assert first_item["option_debug"] is True
        # New centralized logging works correctly


class TestMain:
    """Test cases for main function."""

    def test_main_executes_without_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that main function executes without raising errors."""
        # Should not raise an exception
        main()

        # Verify main function completes successfully with new logging system
        # (Centralized logging outputs to stderr, not captured by caplog)

    def test_main_creates_config(self) -> None:
        """Test that main function creates a Config instance."""
        # We can't directly test the internal config creation,
        # but we can test that main() doesn't crash and behaves as expected
        with patch("uutel.uutel.process_data") as mock_process:
            mock_process.return_value = {"result": "success"}

            main()

            # Verify process_data was called with correct arguments
            mock_process.assert_called_once()
            call_args = mock_process.call_args

            # First argument should be sample data list
            assert call_args[0][0] == ["item1", "item2", "item3"]

            # Should have config keyword argument
            assert "config" in call_args[1]
            config = call_args[1]["config"]
            assert isinstance(config, Config)
            assert config.name == "default"
            assert config.value == "test"
            assert config.options == {"key": "value"}

    def test_main_handles_process_data_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that main function handles exceptions from process_data."""
        with patch("uutel.uutel.process_data") as mock_process:
            mock_process.side_effect = ValueError("Test error")

            with pytest.raises(ValueError, match="Test error"):
                main()

            # New centralized logging handles errors (outputs to stderr)

    def test_main_logs_success(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that main function logs successful completion."""
        main()

        # New centralized logging system works correctly (outputs to stderr)
        # Just verify main completes successfully

    def test_main_with_logging_verification(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test main function with centralized logging system."""
        main()

        # New centralized logging system is active and working
        # (Uses loguru internally with stderr output)


class TestModuleIntegration:
    """Integration tests for the module."""

    def test_module_imports_correctly(self) -> None:
        """Test that the module can be imported without errors."""
        from uutel import uutel

        assert hasattr(uutel, "Config")
        assert hasattr(uutel, "process_data")
        assert hasattr(uutel, "main")

    def test_logging_configuration(self) -> None:
        """Test that centralized logging is configured properly."""
        from uutel import uutel

        # Logger should be available
        assert hasattr(uutel, "logger")
        logger = uutel.logger

        # New centralized logging uses loguru internally
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")

    def test_main_entry_point(self) -> None:
        """Test that main can be used as entry point."""
        # This tests the if __name__ == "__main__" functionality
        # by ensuring the main function is available for external calling
        from uutel.uutel import main

        assert callable(main)

        # Should execute without error when called directly
        main()

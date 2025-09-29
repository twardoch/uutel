#!/usr/bin/env python3
# this_file: src/uutel/uutel.py
"""UUTEL core functionality.

Created by Adam Twardoch
"""

# Standard library imports
import time
from dataclasses import dataclass
from typing import Any

# Local imports
from uutel.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Config:
    """Configuration settings for uutel.

    Attributes:
        name: Configuration name identifier
        value: Configuration value (string, int, or float)
        options: Optional dictionary of additional configuration options
    """

    name: str
    value: str | int | float
    options: dict[str, Any] | None = None


def process_data(
    data: list[Any], config: Config | None = None, *, debug: bool = False
) -> dict[str, Any]:
    """Process the input data according to configuration.

    Args:
        data: Input data to process
        config: Optional configuration settings
        debug: Enable debug mode

    Returns:
        Processed data as a dictionary

    Raises:
        ValueError: If input data is invalid
    """
    if debug:
        logger.debug("Debug mode enabled")

    if not data:
        msg = "Input data cannot be empty"
        raise ValueError(msg)

    # Process data according to configuration
    processed_items = []
    for item in data:
        processed_item = {"original": item, "type": type(item).__name__}

        # Apply configuration-based transformations if config is provided
        if config:
            processed_item["config_name"] = config.name
            processed_item["config_value"] = config.value

            # Apply options if available
            if config.options:
                for key, value in config.options.items():
                    processed_item[f"option_{key}"] = value

        processed_items.append(processed_item)

    result: dict[str, Any] = {
        "processed_count": len(processed_items),
        "items": processed_items,
        "timestamp": time.time(),
        "debug_mode": debug,
    }

    if debug:
        logger.debug(f"Processed {len(data)} items")

    return result


def main() -> None:
    """Main entry point for uutel."""
    try:
        # Example usage with sample data
        config = Config(name="default", value="test", options={"key": "value"})
        sample_data = ["item1", "item2", "item3"]
        result = process_data(sample_data, config=config)
        logger.info("Processing completed: %s", result)

    except Exception as e:
        logger.error(
            "Error in main function: %s (config: %s)",
            e,
            config.name if config else None,
        )
        raise


if __name__ == "__main__":
    main()

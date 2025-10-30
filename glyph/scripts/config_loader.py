#!/usr/bin/env python3
"""
Configuration loader for Glyph rendering system.
Implements priority system: CLI args > Instance config > Dataset YAML config
"""

import os
import yaml
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Loads and merges rendering configurations with priority:
    1. Instance-specific config (from JSON data) - HIGHEST PRIORITY
    2. CLI arguments (from command-line)
    3. Dataset-level config (from YAML file) - LOWEST PRIORITY (default values)
    """

    def __init__(self, yaml_path: str = None):
        """
        Initialize config loader.

        Args:
            yaml_path: Path to dataset_configs.yaml file
        """
        if yaml_path is None:
            # Default to config directory relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_path = os.path.join(script_dir, '../config/dataset_configs.yaml')

        self.yaml_path = yaml_path
        self.dataset_configs = self._load_yaml_configs()

    def _load_yaml_configs(self) -> Dict[str, Any]:
        """Load dataset configurations from YAML file."""
        if not os.path.exists(self.yaml_path):
            print(f"Warning: Dataset config file not found: {self.yaml_path}")
            print("Will use instance configs or CLI args only.")
            return {}

        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                configs = yaml.safe_load(f)
                print(f"Loaded dataset configs from: {self.yaml_path}")
                return configs or {}
        except Exception as e:
            print(f"Error loading YAML config: {e}")
            return {}

    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific dataset.

        Args:
            dataset_name: Name of the dataset (e.g., 'ruler', 'mrcr', 'longbench')

        Returns:
            Dictionary with dataset configuration, or empty dict if not found
        """
        return self.dataset_configs.get(dataset_name, {}).copy()

    def merge_configs(
        self,
        dataset_name: str,
        instance_config: Optional[Dict[str, Any]] = None,
        cli_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Merge configurations following priority system.

        Priority (highest to lowest):
        1. Instance config (from JSON data) - HIGHEST
        2. CLI arguments (cli_overrides)
        3. Dataset YAML config - LOWEST (defaults)

        Args:
            dataset_name: Name of the dataset
            instance_config: Configuration from data instance (JSON)
            cli_overrides: Configuration from command-line arguments

        Returns:
            Merged configuration dictionary
        """
        # Start with dataset-level config (lowest priority - defaults)
        merged = self.get_dataset_config(dataset_name)

        # Apply CLI overrides (middle priority)
        if cli_overrides:
            merged.update({k: v for k, v in cli_overrides.items() if v is not None})

        # Apply instance-specific config (highest priority - overrides everything)
        if instance_config:
            merged.update({k: v for k, v in instance_config.items() if v is not None})

        return merged


def parse_cli_overrides(args) -> Dict[str, Any]:
    """
    Extract CLI arguments that should override configs.

    Args:
        args: argparse.Namespace object from parse_args()

    Returns:
        Dictionary of non-None CLI arguments
    """
    cli_overrides = {}

    # List of possible CLI override parameters
    param_map = {
        'page_size': 'page-size',
        'margin_x': 'margin-x',
        'margin_y': 'margin-y',
        'font_path': 'font-path',
        'font_size': 'font-size',
        'line_height': 'line-height',
        'page_bg_color': 'page-bg-color',
        'font_color': 'font-color',
        'para_bg_color': 'para-bg-color',
        'para_border_color': 'para-border-color',
        'first_line_indent': 'first-line-indent',
        'left_indent': 'left-indent',
        'right_indent': 'right-indent',
        'alignment': 'alignment',
        'space_before': 'space-before',
        'space_after': 'space-after',
        'border_width': 'border-width',
        'border_padding': 'border-padding',
        'horizontal_scale': 'horizontal-scale',
        'dpi': 'dpi',
        'auto_crop_last_page': 'auto-crop-last-page',
        'auto_crop_width': 'auto-crop-width',
        'newline_markup': 'newline-markup'
    }

    for arg_name, config_key in param_map.items():
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value is not None:
                cli_overrides[config_key] = value

    return cli_overrides


def add_dataset_arg(parser):
    """
    Add --dataset argument to argparse parser.

    Args:
        parser: argparse.ArgumentParser object
    """
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name for loading preset config (e.g., 'ruler', 'mrcr', 'longbench')"
    )
    return parser


# Example usage:
if __name__ == '__main__':
    # Test the config loader
    loader = ConfigLoader()

    # Test ruler config
    ruler_config = loader.get_dataset_config('ruler')
    print("\nRuler config:")
    print(ruler_config)

    # Test config merging with priority: Instance > CLI > YAML
    instance_config = {'dpi': 120, 'font-size': 10}
    cli_config = {'dpi': 96, 'margin-x': 15}

    merged = loader.merge_configs('ruler', instance_config, cli_config)
    print("\nMerged config (Priority: Instance > CLI > YAML):")
    print(f"DPI: {merged['dpi']} (should be 120 from instance - HIGHEST)")
    print(f"Font size: {merged['font-size']} (should be 10 from instance)")
    print(f"Margin-x: {merged['margin-x']} (should be 15 from CLI)")
    print(f"Margin-y: {merged['margin-y']} (should be 10 from YAML default)")

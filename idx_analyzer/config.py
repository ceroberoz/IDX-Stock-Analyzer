"""
Configuration management for IDX Analyzer.

Supports TOML configuration files with environment-specific overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .exceptions import ConfigurationError

DEFAULT_CONFIG_PATHS = [
    Path.home() / ".config" / "idx-analyzer" / "config.toml",
    Path.home() / ".idx-analyzer.toml",
    Path.cwd() / "idx-analyzer.toml",
]


@dataclass
class ChartConfig:
    """Chart generation settings"""

    dpi: int = 150
    width: int = 16
    height: int = 10
    style: str = "default"
    show_grid: bool = True
    default_output_dir: Optional[str] = None


@dataclass
class AnalysisConfig:
    """Analysis calculation settings"""

    default_period: str = "6mo"
    rsi_window: int = 14
    sma_windows: list[int] = field(default_factory=lambda: [20, 50, 200])
    bb_window: int = 20
    bb_std: float = 2.0
    vp_bins: int = 50


@dataclass
class NetworkConfig:
    """Network and API settings"""

    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    use_cache: bool = True
    cache_ttl: int = 300


@dataclass
class DisplayConfig:
    """Display and output settings"""

    default_export_format: Optional[str] = None
    color_output: bool = True
    verbose: bool = False


@dataclass
class Config:
    """Main configuration container"""

    chart: ChartConfig = field(default_factory=ChartConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create Config from dictionary"""
        chart_data = data.get("chart", {})
        analysis_data = data.get("analysis", {})
        network_data = data.get("network", {})
        display_data = data.get("display", {})

        return cls(
            chart=ChartConfig(**chart_data),
            analysis=AnalysisConfig(**analysis_data),
            network=NetworkConfig(**network_data),
            display=DisplayConfig(**display_data),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert Config to dictionary"""
        result: dict[str, Any] = {}
        for f in fields(self):
            section = getattr(self, f.name)
            section_dict: dict[str, Any] = {}
            for sf in fields(section):
                value = getattr(section, sf.name)
                if value is not None:
                    section_dict[sf.name] = value
            if section_dict:
                result[f.name] = section_dict
        return result


def find_config_file(explicit_path: Optional[Path] = None) -> Optional[Path]:
    """Find configuration file in standard locations"""
    if explicit_path:
        if explicit_path.exists():
            return explicit_path
        raise ConfigurationError(
            f"Config file not found: {explicit_path}",
            "Check the file path and try again",
        )

    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            return path

    return None


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from TOML file"""
    config_path: Optional[Path] = None
    if path:
        config_path = Path(path)

    found_path = find_config_file(config_path)

    if not found_path:
        return Config()

    try:
        import tomllib

        with open(found_path, "rb") as f:
            data = tomllib.load(f)
    except ImportError:
        try:
            import tomli as tomllib

            with open(found_path, "rb") as f:
                data = tomllib.load(f)
        except ImportError:
            raise ConfigurationError(
                "TOML parser not available",
                "Install Python 3.11+ or pip install tomli",
            )
    except FileNotFoundError:
        raise ConfigurationError(f"Config file not found: {found_path}")
    except Exception as e:
        raise ConfigurationError(
            f"Failed to parse config file: {found_path}",
            str(e),
        )

    return Config.from_dict(data)


def save_config(config: Config, path: Optional[Union[str, Path]] = None) -> Path:
    """Save configuration to TOML file"""
    if path:
        config_path = Path(path)
    else:
        config_path = DEFAULT_CONFIG_PATHS[1]

    try:
        import tomli_w
    except ImportError:
        raise ConfigurationError(
            "Cannot save config: tomli-w not installed",
            "Install with: pip install tomli-w",
        )

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "wb") as f:
        tomli_w.dump(config.to_dict(), f)

    return config_path


def create_default_config(path: Optional[Union[str, Path]] = None) -> Path:
    """Create a default configuration file"""
    config = Config()
    return save_config(config, path)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance"""
    global _config
    _config = config


def reset_config() -> None:
    """Reset global configuration to defaults"""
    global _config
    _config = Config()

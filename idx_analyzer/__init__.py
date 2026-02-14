"""
IDX Stock Analyzer - Indonesian Stock Market Analysis Tool
"""

__version__ = "1.0.0"
__author__ = "IDX Analyzer"

from .analyzer import IDXAnalyzer, AnalysisResult, SupportResistance
from .config import Config, load_config, save_config, create_default_config
from .exceptions import (
    IDXAnalyzerError,
    ConfigurationError,
    DataFetchError,
    NetworkError,
    InvalidTickerError,
    InsufficientDataError,
    AnalysisError,
    ChartError,
    ExportError,
    format_error_for_user,
)

__all__ = [
    "IDXAnalyzer",
    "AnalysisResult",
    "SupportResistance",
    "Config",
    "load_config",
    "save_config",
    "create_default_config",
    "IDXAnalyzerError",
    "ConfigurationError",
    "DataFetchError",
    "NetworkError",
    "InvalidTickerError",
    "InsufficientDataError",
    "AnalysisError",
    "ChartError",
    "ExportError",
    "format_error_for_user",
]
